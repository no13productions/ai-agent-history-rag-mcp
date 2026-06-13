"""Async embedding wrapper with pluggable embedding providers.

Provider "openai" works with any server implementing the OpenAI /v1/embeddings endpoint:
- Ollama (http://localhost:11434/v1)
- vLLM (http://localhost:8000/v1)
- text-embeddings-inference
- OpenAI API
- LiteLLM
- etc.

Provider "vertex" uses Vertex AI prediction endpoints. The default Vertex model is
gemini-embedding-001 with 3072-dimensional output.
"""

import asyncio
import logging
import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx

from claude_history_rag.config import settings

logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 2
RETRY_BACKOFF_BASE = 0.5  # seconds
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=5.0)
VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# Regex patterns for text sanitization
# Match long runs of repeated characters (4+ of the same char)
REPEATED_CHARS_PATTERN = re.compile(r"(.)\1{3,}")
# Match sequences that look like binary/hex data
BINARY_PATTERN = re.compile(r"(?:[0-9a-fA-F]{2}[\s:,]?){8,}")
# Match base64-like long strings (40+ chars of base64 alphabet without spaces)
BASE64_PATTERN = re.compile(r"[A-Za-z0-9+/=]{40,}")


def redact_url(url: str) -> str:
    """Remove credentials from URLs before logging or status output."""
    if not url:
        return url
    parsed = urlsplit(url)
    if not parsed.username and not parsed.password:
        return url
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))


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


class EmbeddingProvider(Protocol):
    """Backend interface for text embedding providers."""

    provider_name: str
    model_name: str
    dimension: int | None

    def initialize(self) -> None:
        """Initialize provider resources."""
        ...

    def embed_sync(self, texts: list[str], task_type: str | None = None) -> list[list[float]]:
        """Synchronously embed text in provider-specific format."""
        ...

    def shutdown(self) -> None:
        """Release provider resources."""
        ...


class OpenAICompatibleEmbeddingProvider:
    """Embedding provider for OpenAI-compatible /v1/embeddings APIs."""

    provider_name = "openai"

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        dimension: int | None = None,
    ):
        self.base_url = (base_url or settings.embedding_base_url).rstrip("/")
        self.model_name = model_name or settings.embedding_model
        self.api_key = api_key if api_key is not None else settings.embedding_api_key
        self.dimension = dimension or settings.embedding_dimension
        self._client: httpx.Client | None = None

    def initialize(self) -> None:
        """Initialize HTTP client."""
        if self._client is not None:
            return
        self._client = httpx.Client(
            timeout=HTTP_TIMEOUT,
            limits=httpx.Limits(
                max_connections=4,
                max_keepalive_connections=2,
                keepalive_expiry=30.0,
            ),
        )
        logger.info(
            "Embedding provider initialized: provider=%s url=%s model=%s",
            self.provider_name,
            redact_url(self.base_url),
            self.model_name,
        )

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def embed_sync(self, texts: list[str], task_type: str | None = None) -> list[list[float]]:
        """Embed using the OpenAI embeddings API format."""
        del task_type
        if self._client is None:
            raise RuntimeError("HTTP client not initialized")

        url = f"{self.base_url}/embeddings"
        payload: dict[str, object] = {
            "model": self.model_name,
            "input": texts,
        }
        if self.dimension is not None and settings.openai_embedding_send_dimensions:
            payload["dimensions"] = self.dimension

        try:
            response = self._client.post(
                url,
                json=payload,
                headers=self._get_headers(),
            )
            response.raise_for_status()
            data = response.json()

            embeddings_data = data.get("data", [])
            if not isinstance(embeddings_data, list) or not all(
                isinstance(item, dict) for item in embeddings_data
            ):
                raise RuntimeError("Invalid embedding response format")
            has_any_index = any("index" in item for item in embeddings_data)
            has_all_indexes = all("index" in item for item in embeddings_data)
            if len(texts) > 1 and not has_all_indexes:
                raise RuntimeError("Invalid embedding response format")
            if has_any_index:
                if not has_all_indexes:
                    raise RuntimeError("Invalid embedding response format")
                indexes = [item.get("index") for item in embeddings_data]
                if not all(isinstance(index, int) for index in indexes) or sorted(indexes) != list(
                    range(len(texts))
                ):
                    raise RuntimeError("Invalid embedding response format")
                embeddings_data.sort(key=lambda x: x["index"])
            embeddings = []
            for item in embeddings_data:
                embedding = item.get("embedding")
                if not isinstance(embedding, list):
                    raise RuntimeError("Invalid embedding response format")
                embeddings.append(embedding)

            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
                )

            return embeddings

        except httpx.HTTPStatusError as e:
            logger.error(
                "Embedding API error: provider=%s model=%s status=%s url=%s",
                self.provider_name,
                self.model_name,
                e.response.status_code,
                redact_url(str(e.request.url)),
            )
            raise RuntimeError(f"Embedding API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            request_url = redact_url(str(e.request.url)) if e.request else "<unknown>"
            logger.error(
                "Embedding request failed: provider=%s model=%s error_type=%s url=%s",
                self.provider_name,
                self.model_name,
                type(e).__name__,
                request_url,
            )
            raise RuntimeError(f"Embedding request failed: {type(e).__name__}") from e
        except (KeyError, IndexError) as e:
            logger.error("Invalid embedding response format: %s", e)
            raise RuntimeError("Invalid embedding response format") from e

    def shutdown(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None


class VertexAIEmbeddingProvider:
    """Embedding provider for Vertex AI gemini-embedding-001."""

    provider_name = "vertex"

    def __init__(
        self,
        project: str | None = None,
        location: str | None = None,
        model_name: str | None = None,
        dimension: int | None = None,
        auto_truncate: bool | None = None,
    ):
        self.project = project if project is not None else settings.vertex_project
        self.location = location or settings.vertex_location
        self.model_name = model_name or settings.embedding_model
        self.dimension = dimension or settings.embedding_dimension
        self.auto_truncate = (
            settings.vertex_auto_truncate if auto_truncate is None else auto_truncate
        )
        self._client: httpx.Client | None = None
        self._credentials = None
        self._endpoint = ""

    def initialize(self) -> None:
        """Initialize Vertex AI REST client."""
        if self._client is not None:
            return

        project = self.project
        from claude_history_rag.gcp_auth import default_project_and_credentials

        resolved_project, credentials = default_project_and_credentials([VERTEX_SCOPE])
        if not project:
            project = resolved_project
        if not project:
            raise RuntimeError(
                "Vertex project is not configured. Set CLAUDE_HISTORY_RAG_VERTEX_PROJECT "
                "or configure Application Default Credentials with a project."
            )

        self.project = project
        self._credentials = credentials
        self._endpoint = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project}/locations/{self.location}/publishers/google/"
            f"models/{self.model_name}:predict"
        )
        self._client = httpx.Client(timeout=HTTP_TIMEOUT)
        logger.info(
            "Embedding provider initialized: provider=%s project=%s location=%s model=%s dim=%s",
            self.provider_name,
            self.project,
            self.location,
            self.model_name,
            self.dimension,
        )

    def _auth_headers(self) -> dict[str, str]:
        """Create authenticated headers for Vertex REST calls."""
        if self._credentials is None:
            raise RuntimeError("Vertex credentials not initialized")
        from google.auth.transport.requests import Request

        if not self._credentials.valid:
            self._credentials.refresh(Request())
        headers = {"Content-Type": "application/json"}
        self._credentials.apply(headers)
        return headers

    def embed_sync(self, texts: list[str], task_type: str | None = None) -> list[list[float]]:
        """Embed using Vertex AI.

        gemini-embedding-001 supports one input per request, so batch requests are
        fanned out here while preserving the public provider interface.
        """
        if self._client is None:
            raise RuntimeError("Vertex embedding client not initialized")

        vectors: list[list[float]] = []
        for text in texts:
            instance: dict[str, object] = {"content": text}
            if task_type:
                instance["task_type"] = task_type
            parameters: dict[str, object] = {"autoTruncate": self.auto_truncate}
            if self.dimension is not None:
                parameters["outputDimensionality"] = self.dimension
            try:
                response = self._client.post(
                    self._endpoint,
                    json={"instances": [instance], "parameters": parameters},
                    headers=self._auth_headers(),
                )
            except httpx.HTTPError as e:
                logger.error(
                    "Vertex embedding transport error: provider=%s project=%s "
                    "location=%s model=%s error=%s",
                    self.provider_name,
                    self.project,
                    self.location,
                    self.model_name,
                    type(e).__name__,
                )
                raise RuntimeError(f"Vertex embedding transport error: {type(e).__name__}") from e
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error(
                    "Vertex embedding API error: provider=%s project=%s location=%s "
                    "model=%s status=%s",
                    self.provider_name,
                    self.project,
                    self.location,
                    self.model_name,
                    e.response.status_code,
                )
                raise RuntimeError(f"Vertex embedding API error: {e.response.status_code}") from e
            try:
                data = response.json()
            except ValueError as e:
                logger.error(
                    "Vertex embedding response JSON error: provider=%s project=%s "
                    "location=%s model=%s error=%s",
                    self.provider_name,
                    self.project,
                    self.location,
                    self.model_name,
                    type(e).__name__,
                )
                raise RuntimeError("Invalid Vertex embedding JSON response") from e
            predictions = data.get("predictions", [])
            if len(predictions) != 1:
                logger.error(
                    "Vertex embedding prediction count mismatch: provider=%s project=%s "
                    "location=%s model=%s count=%s",
                    self.provider_name,
                    self.project,
                    self.location,
                    self.model_name,
                    len(predictions),
                )
                raise RuntimeError(
                    f"Vertex embedding count mismatch: expected 1, got {len(predictions)}"
                )
            embedding = predictions[0].get("embeddings", {})
            values = embedding.get("values")
            if not isinstance(values, list):
                logger.error(
                    "Vertex embedding response shape error: provider=%s project=%s "
                    "location=%s model=%s",
                    self.provider_name,
                    self.project,
                    self.location,
                    self.model_name,
                )
                raise RuntimeError("Invalid Vertex embedding response format")
            vectors.append(values)
        return vectors

    def shutdown(self) -> None:
        """Close REST client."""
        if self._client is not None:
            self._client.close()
            self._client = None
        self._credentials = None


def create_embedding_provider(
    provider_name: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
) -> EmbeddingProvider:
    """Create an embedding provider from settings."""
    provider = (provider_name or settings.embedding_provider).lower()
    if provider == "openai":
        return OpenAICompatibleEmbeddingProvider(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
        )
    if provider == "vertex":
        return VertexAIEmbeddingProvider(model_name=model_name)
    raise ValueError(f"Unsupported embedding provider: {provider}")


class AsyncEmbedder:
    """Async embedding facade with pluggable providers."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        provider: EmbeddingProvider | None = None,
        provider_name: str | None = None,
    ):
        self.provider = provider or create_embedding_provider(
            provider_name=provider_name,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
        )
        self.base_url = getattr(self.provider, "base_url", "")
        self.model_name = self.provider.model_name
        self.provider_name = self.provider.provider_name
        self.dimension = self.provider.dimension
        self._client_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._shutdown = False
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure provider resources are initialized."""
        if self._initialized:
            return

        with self._client_lock:
            if not self._initialized:
                self.provider.initialize()
                self.base_url = getattr(self.provider, "base_url", "")
                self.model_name = self.provider.model_name
                self.provider_name = self.provider.provider_name
                self.dimension = self.provider.dimension
                self._initialized = True

    def _embed_sync(self, texts: list[str], task_type: str | None = None) -> list[list[float]]:
        """Synchronous embedding through the configured provider.

        Args:
            texts: List of texts to embed
            task_type: Optional provider-specific embedding task type

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If embedding fails
        """
        if self._shutdown:
            raise RuntimeError("Cannot embed after shutdown")
        return self.provider.embed_sync(texts, task_type=task_type)

    async def embed(self, texts: list[str], task_type: str | None = None) -> list[list[float]]:
        """Embed texts asynchronously."""
        if self._shutdown:
            raise RuntimeError("Cannot embed after shutdown")

        await self._ensure_initialized()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._embed_sync,
            texts,
            task_type,
        )

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query.

        For Nomic models, we add search_query: prefix.
        """
        # Nomic requires search_query prefix for queries
        if "nomic" in self.model_name.lower():
            query = f"search_query: {query}"
        task_type = settings.vertex_query_task_type if self.provider_name == "vertex" else None
        embeddings = await self.embed([query], task_type=task_type)
        return embeddings[0]

    async def embed_documents(self, docs: list[str]) -> list[list[float]]:
        """Embed documents.

        For Nomic models, we add search_document: prefix.
        """
        # Nomic requires search_document prefix for documents
        if "nomic" in self.model_name.lower():
            docs = [f"search_document: {doc}" for doc in docs]
        task_type = settings.vertex_document_task_type if self.provider_name == "vertex" else None
        return await self.embed(docs, task_type=task_type)

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
                            f"(attempt {attempt + 1}/{MAX_RETRIES}), retrying: "
                            f"{type(e).__name__}; provider={settings.embedding_provider} "
                            f"model={settings.embedding_model} chunks={len(valid_chunks)}"
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
            self.provider.shutdown()

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
