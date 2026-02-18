import os
from pathlib import Path

import nest_asyncio
from dotenv import load_dotenv

from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status

import fitz  # PyMuPDF

nest_asyncio.apply()
load_dotenv()


async def initialize_rag(working_dir: str = "./rag_storage") -> LightRAG:
    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        graph_storage="Neo4JStorage",
        vector_storage="FaissVectorDBStorage",
        chunk_token_size=1500,
        chunk_overlap_token_size=300
    )
    await rag.initialize_storages()
    initialize_share_data()
    await initialize_pipeline_status()
    return rag


async def index_data(rag: LightRAG, file_path: str) -> None:
    """
    Indexa UM arquivo (PDF ou texto) no LightRAG.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    if file_path.lower().endswith(".pdf"):
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    if not text.strip():
        print(f"Aviso: {file_path} vazio ou não pôde ser lido.")
        return

    await rag.ainsert(text)
    print(f"Sucesso: {file_path} indexado com sucesso.")


async def index_file(rag: LightRAG, path: str) -> int:
    """
    Indexa:
    - 1 arquivo, se 'path' for arquivo
    - todos os PDFs, se 'path' for diretório
    Retorna quantidade de arquivos indexados.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {path}")

    # Caso 1: arquivo único
    if p.is_file():
        await index_data(rag, str(p))
        return 1

    # Caso 2: diretório -> todos os PDFs (recursivo)
    pdf_files = sorted([f for f in p.rglob("*.pdf") if f.is_file()])

    if not pdf_files:
        print(f"Aviso: nenhum PDF encontrado em {path}")
        return 0

    count = 0
    for pdf in pdf_files:
        await index_data(rag, str(pdf))
        count += 1

    print(f"Total de PDFs indexados: {count}")
    return count


