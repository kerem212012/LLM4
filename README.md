# LLM4

A small retrieval-augmented generation demo using Anthropic and ChromaDB.

## Setup

1. Create a Python environment:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```bash
python -m pip install -U pip
python -m pip install .
```

3. Create a `.env` file with:
```env
API_KEY=your_anthropic_api_key
MODEL=claude-sonnet-4-6
MAX_TOKENS=2048
```

## Running

### Use the CLI
```bash
python main.py
```

Supported commands:
- Type a question directly to ask the chat bot.
- `/search <query>` to search indexed documents.
- `/index` to rebuild the document index.
- `/clear` to clear chat history.
- `/quit` or `/exit` to exit.

## Documents

Put text files in `docs/` to include them in the vector store index.
