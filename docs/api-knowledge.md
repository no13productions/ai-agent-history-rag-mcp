# API Knowledge (Updated December 2025)

Research sources for agents to reference.

## MCP Python SDK

**Version**: mcp 1.24.0 (Dec 12, 2025)
**Source**: https://pypi.org/project/mcp/, https://github.com/modelcontextprotocol/python-sdk

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ai-agent-history-rag")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

- Python >=3.10 required
- Two FastMCP implementations exist:
  - Official: `from mcp.server.fastmcp import FastMCP` (use this)
  - jlowin's: `from fastmcp import FastMCP` (more features, separate package)

## LanceDB

**Version**: lancedb 0.25.3 (Nov 7, 2025)
**Source**: https://pypi.org/project/lancedb/, https://lancedb.com/

```python
from lancedb.rerankers import RRFReranker

# Hybrid search with RRF reranking
result = tbl.search("hello", query_type="hybrid").rerank(reranker=RRFReranker()).to_list()
```

- RRFReranker is default for hybrid search
- `normalize` param: "rank" or "score"

## fastembed (historical reference only)

**Version**: fastembed 0.7.4 (Dec 5, 2025)
**Source**: https://pypi.org/project/fastembed/, https://qdrant.github.io/fastembed/

The current server/standalone embedding path does not use `fastembed` in-process.
It uses an external OpenAI-compatible `/v1/embeddings` endpoint, Vertex AI REST,
or Spanner-native `ML.PREDICT` against a registered Vertex embedding model.

```python
from fastembed import TextEmbedding

model = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5")
embeddings = list(model.embed(documents))
```

- nomic-ai/nomic-embed-text-v1.5 is supported (768 dims, 0.520 GB, Apache 2.0)
- ONNX-based, no PyTorch needed

## Nomic Embed Text v1.5

**Source**: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5

**Prefixes (REQUIRED)**:
- Documents: `search_document: <text>`
- Queries: `search_query: <text>`
- Clustering: `clustering: <text>`
- Classification: `classification: <text>`

**Matryoshka dimensions**:
| Dims | MTEB Score |
|------|-----------|
| 768  | 62.28     |
| 512  | 61.96     |
| 256  | 61.04     |
| 128  | 59.34     |
| 64   | 56.10     |

Max sequence: 8192 tokens

## watchfiles

**Version**: watchfiles 1.1.1 (Oct 14, 2025)
**Source**: https://pypi.org/project/watchfiles/, https://watchfiles.helpmanual.io/

```python
from watchfiles import awatch

async for changes in awatch('./my/dir',
                            watch_filter=lambda c, p: p.endswith(".jsonl"),
                            debounce=5000,  # ms
                            recursive=True):
    for change_type, path in changes:
        print(change_type, path)
```

**awatch parameters**:
- `watch_filter`: callable to filter changes
- `debounce`: ms to group changes (default 1600)
- `recursive`: watch subdirs (default True)
- `stop_event`: asyncio.Event to stop watching
- Requires: anyio>=3.0.0, Python 3.9-3.14
