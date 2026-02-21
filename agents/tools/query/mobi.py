# app_agents.py
from __future__ import annotations

from agents.plugins.ingestion import initialize_rag, index_file
from agents.plugins.retrieve import run_async_query


async def query_fiatmobi_database(query: str, mode: str = "mix") -> str:
    """Consulta a base de dados LightRAG do manual do Fiat Mobi."""
    rag = await initialize_rag()
    total = await index_file(rag, "../database/documents/manual-fiatmobi.pdf")
    if total == 0:
        return "Nenhum PDF do Fiat Mobi foi indexado."
    resp = await run_async_query(rag, query, mode)
    return f"[Fiat Mobi] Resultado para: {query}\nResposta: {resp}"
