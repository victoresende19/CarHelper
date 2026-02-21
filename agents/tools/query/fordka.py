# app_agents.py
from __future__ import annotations

from agents.plugins.ingestion import initialize_rag, index_file
from agents.plugins.retrieve import run_async_query

# import os

# from pathlib import Path

# HERE = Path(__file__).resolve()

# # sobe até encontrar a pasta "agents"
# AGENTS_DIR = next(p for p in HERE.parents if p.name == "agents")

# PDF_PATH = AGENTS_DIR / "database" / "documents" / "manual-fordka.pdf"

# if not PDF_PATH.exists():
#     raise FileNotFoundError(f"Caminho não encontrado: {PDF_PATH}")

async def query_fordka_database(query: str, mode: str = "mix") -> str:
    """Consulta a base de dados LightRAG do manual do Ford KA."""
    rag = await initialize_rag()
    total = await index_file(rag, "../database/documents/manual-fordka.pdf")
    if total == 0:
        return "Nenhum PDF do Ford KA foi indexado."
    resp = await run_async_query(rag, query, mode)
    return f"[Ford KA] Resultado para: {query}\nResposta: {resp}"

