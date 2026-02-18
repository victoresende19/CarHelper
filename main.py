from utils.ingestion import initialize_rag, index_file
from utils.retrieve import run_async_query
from dotenv import load_dotenv
import asyncio

load_dotenv()


async def main(question: str, mode, data_path: str = "data/manual-fordka.pdf") -> None:
    """
    1. Initialize RAG
    2. Index file (open file, read file, chunking, stream each chunk to both vector store and knowldege graph) all being done by rag.ainsert().
    3. Run async queries
    """
    rag = await initialize_rag()
    await index_file(rag, data_path) # this function wait here until all files be

    # run query
    resp_async = await run_async_query(rag, question, mode)
    print("\n===== Query Result =====\n")
    print(resp_async)


if __name__ == "__main__":
    question = "Em qual casos a garantia Ford não cobre o veículo?"
    mode = "mix"
    asyncio.run(main(question, mode))