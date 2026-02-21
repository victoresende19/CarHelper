import os
from typing import List
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.base_tool import BaseTool
from google.adk.agents.readonly_context import ReadonlyContext

from agents.plugins.retrieve import run_async_query
from agents.plugins.ingestion import initialize_rag, index_file

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FORD_KA_PDF = os.path.join(BASE_DIR, "database", "documents", "manual-fordka.pdf")
MOBI_PDF = os.path.join(BASE_DIR, "database", "documents", "manual-mobi.pdf")
RAG_STORAGE_DIR = os.path.join(BASE_DIR, "database", "rag_storage")

class ManualToolset(BaseToolset):
    async def get_tools(self, readonly_context: ReadonlyContext) -> List[BaseTool]:

        # Identificar agente está executando no momento
        agent_name = readonly_context.agent_name

        # Filtros atualizados com os caminhos absolutos
        pdf_filters = {
            "especialista_fordka": [FORD_KA_PDF],
            "especialista_fiatmobi": [MOBI_PDF],
            "especialista_generalista": [
                FORD_KA_PDF, 
                MOBI_PDF
            ]
        }

        async def busca_documentos(query: str) -> str:
            """Consulta a base de dados dos manuais automotivos."""
            pdfs_para_buscar = pdf_filters.get(agent_name, [])

            if not pdfs_para_buscar:
                return "Nenhum manual configurado para este especialista."

            rag = await initialize_rag(RAG_STORAGE_DIR)
            
            # Verifica se o arquivo realmente existe antes de tentar indexar (boa prática)
            for pdf_path in pdfs_para_buscar:
                if not os.path.exists(pdf_path):
                    print(f"ERRO: Arquivo não encontrado -> {pdf_path}")
                    continue
                
                await index_file(rag, pdf_path)

            resp = await run_async_query(rag, query, "mix")
            return f"[{agent_name}] Resultado da busca:\n{resp}"

        # Retorna a função genérica empacotada como ferramenta
        return [FunctionTool(func=busca_documentos)]
