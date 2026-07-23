from pathlib import Path

import chromadb
from anthropic import Anthropic
from rich.console import Console
from rich.table import Table

from config import API_KEY, MAX_TOKENS, MODEL

CHROMA_PATH = "./chroma_db"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using the provided document context. "
    "If the answer is based on the supplied context, cite the source filename in your response."
)


def chunk_text(text, chunk_size=500, overlap=50):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    step = chunk_size - overlap
    chunks = []
    start = 0

    while start < len(cleaned_text):
        end = start + chunk_size
        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned_text):
            break

        start += step

    return chunks


def get_or_create_collection() -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = chroma_client.get_or_create_collection(name="documents", metadata={"hnsw:space": "cosine"})

    return col


def load_documents(folder: str) -> list[dict]:
    docs = []
    for file_path in Path(folder).iterdir():
        if file_path.suffix not in {".txt", ".md"}:
            continue
        text = file_path.read_text(encoding="utf-8")
        if text:
            docs.append({"text": text, "source": str(file_path), "filename": file_path.name})
    return docs


def build_index(documents: list[dict]) -> chromadb.Collection:
    collection = get_or_create_collection()
    existing = set(collection.get(include=[])["ids"])
    texts_to_add, ids_to_add, metadatas_to_add = [], [], []
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['filename']}::chunk_{i}"
            if chunk_id in existing:
                continue
            texts_to_add.append(chunk)
            ids_to_add.append(chunk_id)
            metadatas_to_add.append({"source": doc["source"], "filename": doc["filename"], "chunk_index": i, "total_chunks": len(chunks)})
    if texts_to_add:
        collection.add(documents=texts_to_add, ids=ids_to_add, metadatas=metadatas_to_add)
    return collection

def search(collection, query: str, n_results: int = 3) -> list[dict]:
    total = collection.count()
    if total == 0:
        return []
    n = min(n_results, total)
    results = collection.query(query_texts=[query], n_results=n, include=["documents", "metadatas", "distances"])
    documents, metadatas, distances = results["documents"][0], results["metadatas"][0], results["distances"][0]
    hits = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        hits.append({"text": doc, "filename": meta["filename"], "distance": dist})
    return hits


def show_search_results(results: list[dict]) -> None:
    table = Table(title="Search results")
    table.add_column("File", style="cyan")
    table.add_column("Relevance %", justify="right", style="magenta")
    table.add_column("Fragment", overflow="fold")

    for item in results:
        fragment = item["text"]
        if len(fragment) > 120:
            fragment = fragment[:117] + "..."
        relevance_percent = round((1 - item["distance"]) * 100, 1)
        table.add_row(item["filename"], f"{relevance_percent:.1f}%", fragment)

    Console().print(table)

def build_context(results: list[dict]) -> str:
    if not results:
        return "Контекст из документов: (ничего не найдено)"

    parts = []
    for i, result in enumerate(results, start=1):
        relevance = round((1 - result["distance"]) * 100, 1)
        header = f"Фрагмент {i} | Источник: {result['filename']} | Релевантность: {relevance:.1f}%"
        parts.append(f"{header}\n{result['text']}")
    return "\n".join(parts)


def get_client() -> Anthropic:
    if not API_KEY:
        raise RuntimeError("API_KEY is not set. Populate the environment before running the chat flow.")
    return Anthropic(api_key=API_KEY)


def rag_chat(question: str, history: list[dict], collection) -> tuple[str, list[dict]]:
    search_results = search(collection=collection, query=question, n_results=3)
    context = build_context(search_results)
    system_with_context = SYSTEM_PROMPT + "\n\n" + context

    history.append({"role": "user", "content": question})

    client = get_client()
    console = Console()
    chunks: list[str] = []

    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_with_context,
        messages=history,
    ) as stream:
        for chunk in stream.text_stream:
            if chunk:
                console.print(chunk, end="")
                chunks.append(chunk)

    console.print()
    answer = "".join(chunks)
    history.append({"role": "assistant", "content": answer})
    return answer, history


def handle_command(command: str, history: list[dict], collection) -> tuple[bool, list[dict]]:
    if command.lower() in {"/quit", "/exit"}:
        return True, history

    if command.startswith("/search"):
        query = command[len("/search"):].strip()
        if not query:
            print("Please provide a search query.")
            return False, history
        results = search(collection=collection, query=query, n_results=3)
        show_search_results(results)
        print(build_context(results))
        return False, history

    if command.lower() == "/clear":
        history.clear()
        print("Chat history cleared.")
        return False, history

    if command.lower() == "/index":
        documents = load_documents("docs")
        build_index(documents)
        print(f"Indexed {len(documents)} documents from docs/.")
        return False, history

    answer, history = rag_chat(command, history, collection)
    print()
    print(f"Answer: {answer}")
    return False, history


if __name__ == '__main__':
    collection = build_index(load_documents("docs"))
    history: list[dict] = []
    print("Type a question about the docs, /search <query>, /clear, /index, or /quit")
    while True:
        command = input("> ").strip()
        if not command:
            continue
        should_exit, history = handle_command(command, history, collection)
        if should_exit:
            break

