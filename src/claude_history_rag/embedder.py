"""Async embedding wrapper using OpenAI-compatible API.

Works with any embedding server that implements the OpenAI /v1/embeddings endpoint:
- Ollama (http://localhost:11434/v1)
- vLLM (http://localhost:8000/v1)
- text-embeddings-inference
- OpenAI API
- LiteLLM
- etc.
"""

import asyncio
import logging
import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import httpx

from claude_history_rag.config import settings

logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 2
RETRY_BACKOFF_BASE = 0.5  # seconds
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=5.0)

# Regex patterns for text sanitization
# Match long runs of repeated characters (4+ of the same char)
REPEATED_CHARS_PATTERN = re.compile(r"(.)\1{3,}")
# Match sequences that look like binary/hex data
BINARY_PATTERN = re.compile(r"(?:[0-9a-fA-F]{2}[\s:,]?){8,}")
# Match base64-like long strings (40+ chars of base64 alphabet without spaces)
BASE64_PATTERN = re.compile(r"[A-Za-z0-9+/=]{40,}")


def sanitize_text_for_embedding(text: str) -> str:
    """Sanitize text to prevent NaN errors from embedding models.

    This function removes or replaces problematic characters that can cause
    embedding models (especially bge-m3) to produce NaN values, while
    preserving the semantic content as much as possible.

    Transformations applied:
    1. Normalize Unicode to NFC form
    2. Remove control characters (except newlines/tabs)
    3. Replace unusual Unicode categories (symbols, private use, etc.)
    4. Collapse long repeated character sequences
    5. Truncate binary/hex-looking data
    6. Truncate base64-looking strings

    Args:
        text: Raw text that may contain problematic characters

    Returns:
        Sanitized text safe for embedding
    """
    if not text:
        return text

    # 1. Normalize Unicode to composed form (NFC)
    text = unicodedata.normalize("NFC", text)

    # 2. Remove/replace problematic characters
    cleaned_chars = []
    for char in text:
        # Keep basic ASCII printable and common whitespace
        if char in "\n\t\r" or (32 <= ord(char) < 127):
            cleaned_chars.append(char)
            continue

        # Check Unicode category
        category = unicodedata.category(char)

        # Keep letters, numbers, punctuation, and common marks
        if category[0] in ("L", "N", "P", "M"):
            cleaned_chars.append(char)
        # Replace spaces, separators, control chars, symbols with space
        # This preserves word boundaries while removing problematic chars
        else:
            cleaned_chars.append(" ")

    text = "".join(cleaned_chars)

    # 3. Collapse long repeated characters (e.g., "======" -> "==")
    text = REPEATED_CHARS_PATTERN.sub(r"\1\1", text)

    # 4. Truncate binary/hex-looking sequences
    text = BINARY_PATTERN.sub("[binary data]", text)

    # 5. Truncate base64-looking strings
    text = BASE64_PATTERN.sub("[encoded data]", text)

    # 6. Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    # 7. Strip leading/trailing whitespace
    text = text.strip()

    return text


class AsyncEmbedder:
    """Async wrapper for OpenAI-compatible embedding APIs."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
    ):
        self.base_url = (base_url or settings.embedding_base_url).rstrip("/")
        self.model_name = model_name or settings.embedding_model
        self.api_key = api_key if api_key is not None else settings.embedding_api_key
        self._client: httpx.Client | None = None
        self._client_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._shutdown = False
        self._initialized = False

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _ensure_initialized(self) -> None:
        """Ensure HTTP client is initialized."""
        if self._initialized:
            return

        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(
                    timeout=HTTP_TIMEOUT,
                    limits=httpx.Limits(
                        max_connections=4,
                        max_keepalive_connections=2,
                        keepalive_expiry=30.0,
                    ),
                )
                logger.info(
                    f"Embedding client initialized: {self.base_url}, model: {self.model_name}"
                )

        self._initialized = True

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding using OpenAI-compatible API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If embedding fails
        """
        if self._shutdown:
            raise RuntimeError("Cannot embed after shutdown")
        if self._client is None:
            raise RuntimeError("HTTP client not initialized")

        # OpenAI embeddings API format
        url = f"{self.base_url}/embeddings"
        payload = {
            "model": self.model_name,
            "input": texts,
        }

        try:
            response = self._client.post(
                url,
                json=payload,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()

            # Extract embeddings from response
            # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
            embeddings_data = data.get("data", [])

            # Sort by index to ensure correct order
            embeddings_data.sort(key=lambda x: x.get("index", 0))

            embeddings = [item["embedding"] for item in embeddings_data]

            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
                )

            return embeddings

        except httpx.HTTPStatusError as e:
            logger.error(f"Embedding API error: {e.response.status_code} - {e.response.text[:200]}")
            raise RuntimeError(f"Embedding API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error(f"Embedding request failed: {type(e).__name__}: {e}")
            raise RuntimeError(f"Embedding request failed: {type(e).__name__}") from e
        except (KeyError, IndexError) as e:
            logger.error(f"Invalid embedding response format: {e}")
            raise RuntimeError("Invalid embedding response format") from e

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts asynchronously."""
        if self._shutdown:
            raise RuntimeError("Cannot embed after shutdown")

        await self._ensure_initialized()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._embed_sync,
            texts,
        )

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query.

        For Nomic models, we add search_query: prefix.
        """
        # Nomic requires search_query prefix for queries
        if "nomic" in self.model_name.lower():
            query = f"search_query: {query}"
        embeddings = await self.embed([query])
        return embeddings[0]

    async def embed_documents(self, docs: list[str]) -> list[list[float]]:
        """Embed documents.

        For Nomic models, we add search_document: prefix.
        """
        # Nomic requires search_document prefix for documents
        if "nomic" in self.model_name.lower():
            docs = [f"search_document: {doc}" for doc in docs]
        return await self.embed(docs)

    async def embed_chunks(
        self,
        chunks: list[dict],
        batch_size: int | None = None,
    ) -> list[dict]:
        """Embed chunks in batches and add vectors.

        Args:
            chunks: List of chunk dicts with 'content' field
            batch_size: Batch size for embedding (default from settings)

        Returns:
            Chunks with 'vector' field added (failed chunks excluded)
        """
        batch_size = batch_size or settings.batch_size
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        await self._ensure_initialized()

        successful_chunks = []
        failed_batches = 0
        total_batches = 0

        for i in range(0, len(chunks), batch_size):
            total_batches += 1
            batch = chunks[i : i + batch_size]
            # Validate chunks are dicts and have content
            valid_chunks = []
            texts = []
            for c in batch:
                if not isinstance(c, dict):
                    logger.warning(f"Skipping non-dict chunk at batch {i}")
                    continue
                content = c.get("content", "")
                if not content or not content.strip():
                    logger.warning(f"Skipping empty content chunk: {c.get('id', '?')[:8]}")
                    continue
                valid_chunks.append(c)
                texts.append(content)

            if not texts:
                logger.warning(f"Batch {i}-{i + batch_size} has no valid content, skipping")
                failed_batches += 1
                continue

            batch_failed = False
            # Retry loop only for transient failures (exceptions), not validation errors
            for attempt in range(MAX_RETRIES + 1):
                try:
                    vectors = await self.embed_documents(texts)

                    # Validation checks - don't retry on these failures
                    if len(vectors) != len(valid_chunks):
                        chunk_ids = [c.get("id", "?")[:8] for c in valid_chunks[:3]]
                        logger.error(
                            f"Embedding count mismatch: expected {len(valid_chunks)}, "
                            f"got {len(vectors)}, chunk_ids: {chunk_ids}"
                        )
                        batch_failed = True
                        break  # Don't retry on validation failures

                    # Validate and assign vectors
                    validation_failed = False
                    for chunk, vector in zip(valid_chunks, vectors, strict=True):
                        # Validate vector is a list of floats
                        if not isinstance(vector, list) or not all(
                            isinstance(v, (int, float)) for v in vector
                        ):
                            logger.error(
                                f"Invalid vector type for chunk {chunk.get('id', '?')[:8]}"
                            )
                            validation_failed = True
                            break
                        chunk["vector"] = vector
                        successful_chunks.append(chunk)

                    if validation_failed:
                        batch_failed = True
                        break  # Don't retry on validation failures

                    # Success - exit retry loop
                    break

                except Exception as e:
                    # Only retry on exceptions (transient failures)
                    chunk_ids = [c.get("id", "?")[:8] for c in valid_chunks[:3]]
                    if attempt < MAX_RETRIES:
                        logger.warning(
                            f"Embedding batch {i}-{i + batch_size} failed "
                            f"(attempt {attempt + 1}/{MAX_RETRIES}), retrying: {type(e).__name__}"
                        )
                        await asyncio.sleep(RETRY_BACKOFF_BASE * (attempt + 1))
                    else:
                        logger.warning(
                            f"Batch {i}-{i + batch_size} failed after "
                            f"{MAX_RETRIES + 1} attempts, falling back to individual embedding"
                        )
                        # Fallback: try embedding each chunk individually to salvage good ones
                        individual_success = 0
                        individual_failed = 0
                        sanitized_success = 0
                        for chunk, text in zip(valid_chunks, texts, strict=True):
                            chunk_id = chunk.get("id", "?")[:8]
                            try:
                                vectors = await self.embed_documents([text])
                                if vectors and len(vectors) == 1:
                                    vector = vectors[0]
                                    if isinstance(vector, list) and all(
                                        isinstance(v, (int, float)) for v in vector
                                    ):
                                        chunk["vector"] = vector
                                        successful_chunks.append(chunk)
                                        individual_success += 1
                                        continue
                            except Exception as ind_e:
                                logger.debug(
                                    f"Individual embed failed for {chunk_id}: "
                                    f"{type(ind_e).__name__}, trying sanitized"
                                )
                                # Try with sanitized text
                                try:
                                    sanitized = sanitize_text_for_embedding(text)
                                    if sanitized and sanitized != text:
                                        vectors = await self.embed_documents([sanitized])
                                        if vectors and len(vectors) == 1:
                                            vector = vectors[0]
                                            if isinstance(vector, list) and all(
                                                isinstance(v, (int, float)) for v in vector
                                            ):
                                                chunk["vector"] = vector
                                                successful_chunks.append(chunk)
                                                sanitized_success += 1
                                                logger.info(
                                                    f"Chunk {chunk_id} succeeded after sanitization "
                                                    f"(removed {len(text) - len(sanitized)} chars)"
                                                )
                                                continue
                                except Exception as san_e:
                                    logger.debug(
                                        f"Sanitized embed also failed for {chunk_id}: "
                                        f"{type(san_e).__name__}"
                                    )
                            individual_failed += 1

                        total_saved = individual_success + sanitized_success
                        logger.info(
                            f"Individual fallback: {total_saved} succeeded "
                            f"({individual_success} normal, {sanitized_success} sanitized), "
                            f"{individual_failed} failed out of {len(valid_chunks)} chunks"
                        )
                        # Only count as batch_failed if we couldn't save any chunks
                        batch_failed = total_saved == 0

            if batch_failed:
                failed_batches += 1

        # Log summary of batch processing
        logger.info(
            f"Embedding complete: {len(successful_chunks)}/{len(chunks)} chunks, "
            f"{failed_batches}/{total_batches} batches failed"
        )

        # Raise exception if ALL batches failed
        if total_batches > 0 and failed_batches == total_batches:
            raise RuntimeError(
                f"All {total_batches} embedding batches failed, no chunks successfully embedded"
            )

        return successful_chunks

    @property
    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self._initialized

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool executor and HTTP client.

        This method is idempotent - safe to call multiple times.
        """
        if self._shutdown:
            logger.debug("Embedder already shut down, skipping")
            return

        logger.info("Shutting down embedder")
        self._shutdown = True

        # Close HTTP client
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

        # Shutdown thread pool
        self._executor.shutdown(wait=wait, cancel_futures=True)


# Global embedder instance (lazy initialization)
embedder: AsyncEmbedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> AsyncEmbedder:
    """Get or create the global embedder instance (thread-safe)."""
    global embedder
    # Double-check pattern for thread safety
    if embedder is None:
        with _embedder_lock:
            # Check again after acquiring lock
            if embedder is None:
                embedder = AsyncEmbedder()
    return embedder
