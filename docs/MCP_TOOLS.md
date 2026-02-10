# Anki Atlas MCP Tools

MCP (Model Context Protocol) server exposing Anki Atlas tools for AI agents like Claude Code.

## Installation

The MCP server is included in the main package. Install Anki Atlas and ensure the MCP dependency is available:

```bash
uv sync
```

## Running the Server

### Standalone

```bash
# Via entry point
uv run anki-atlas-mcp

# Or directly
uv run python -m apps.mcp.cli
```

### With MCP Inspector (for testing)

```bash
npx @anthropic-ai/mcp-inspector uv run anki-atlas-mcp
```

## Configuration

### Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "anki-atlas": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/anki-atlas", "anki-atlas-mcp"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/anki_atlas",
        "QDRANT_URL": "http://localhost:6333",
        "EMBEDDING_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### Claude Code

Add to your project's `.mcp.json`:

```json
{
  "servers": {
    "anki-atlas": {
      "command": "uv",
      "args": ["run", "anki-atlas-mcp"],
      "cwd": "/path/to/anki-atlas",
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/anki_atlas",
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

## Available Tools

### ankiatlas_search

Search Anki notes using hybrid semantic and full-text search.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query text |
| `limit` | integer | No | 20 | Maximum results (1-100) |
| `deck_filter` | string[] | No | null | Filter by deck names |
| `tag_filter` | string[] | No | null | Filter by tags |
| `semantic_only` | boolean | No | false | Use only semantic search |
| `fts_only` | boolean | No | false | Use only full-text search |

**Example prompts:**
- "Search my Anki cards for 'mitochondria'"
- "Find all calculus cards in my Math deck"
- "Search for cards tagged 'exam' about derivatives"

### ankiatlas_topic_coverage

Get coverage metrics for a topic in your Anki collection.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `topic_path` | string | Yes | - | Topic path (e.g., 'math/calculus') |
| `include_subtree` | boolean | No | true | Include child topics in metrics |

**Example prompts:**
- "How well do I cover the topic 'biology/genetics'?"
- "Show me coverage stats for my math collection"
- "What's my progress on the 'programming/python' topic?"

### ankiatlas_topic_gaps

Find knowledge gaps in topic coverage.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `topic_path` | string | Yes | - | Root topic path to analyze |
| `min_coverage` | integer | No | 1 | Minimum notes for coverage |

**Example prompts:**
- "What topics am I missing in my chemistry notes?"
- "Find gaps in my language learning cards"
- "Show me undercovered areas in 'history'"

### ankiatlas_duplicates

Find near-duplicate notes in your Anki collection.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `threshold` | float | No | 0.92 | Similarity threshold (0.5-1.0) |
| `max_clusters` | integer | No | 50 | Maximum clusters to return |
| `deck_filter` | string[] | No | null | Filter by deck names |
| `tag_filter` | string[] | No | null | Filter by tags |

**Example prompts:**
- "Find duplicate cards in my collection"
- "Are there any near-duplicates in my vocabulary deck?"
- "Check for redundant flashcards with 95% similarity"

### ankiatlas_sync

Sync an Anki collection to the search index.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `collection_path` | string | Yes | - | Path to collection.anki2 file |
| `run_index` | boolean | No | false | Rebuild vector index after sync |

**Example prompts:**
- "Sync my Anki collection at /path/to/collection.anki2"
- "Update the index from my Anki database and rebuild embeddings"

## Example Agent Workflows

### Comprehensive Collection Review

```
1. First, sync my Anki collection from ~/Library/Application Support/Anki2/User 1/collection.anki2

2. Then find any duplicate cards in my collection with 90% similarity threshold

3. Show me topic coverage for 'medicine' and identify any gaps

4. Search for cards about 'pharmacology' to see what I have
```

### Study Session Preparation

```
1. Show me coverage for 'math/linear-algebra'

2. Find gaps in the linear algebra topic tree

3. Search for weak cards (high lapse rate) about matrices
```

### Collection Cleanup

```
1. Find all duplicates in my 'Vocabulary' deck

2. Search for cards with similar content about 'verb conjugation'

3. Show me notes that might be redundant
```

## Output Format

All tools return markdown-formatted responses optimized for LLM readability:

- **Tables** for search results and gap lists
- **Headings** for section organization
- **Bold** for key metrics
- **Code blocks** for paths and technical identifiers
- **Truncated previews** to keep responses concise

## Error Handling

Tools return user-friendly error messages prefixed with `**Error**:` instead of raising exceptions. Common errors:

- Collection file not found
- Topic not found in taxonomy
- Database connection issues
- Invalid parameters

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `QDRANT_URL` | Yes | Qdrant vector database URL |
| `EMBEDDING_PROVIDER` | No | "openai" or "local" (default: openai) |
| `OPENAI_API_KEY` | If using OpenAI | OpenAI API key for embeddings |
| `EMBEDDING_MODEL` | No | Model name (default: text-embedding-3-small) |
