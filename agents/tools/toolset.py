from typing import List
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.base_tool import BaseTool
from google.adk.agents.readonly_context import ReadonlyContext

from plugins.retrieve import run_async_query
from plugins.ingestion import initialize_rag, index_file

class ManualToolset(BaseToolset):
    async def get_tools(self, readonly_context: ReadonlyContext) -> List[BaseTool]:

        # Identificar agente está executando no momento
        agent_name = readonly_context.agent_name

        # Filtros por dicionário para ficar menos verboso
        pdf_filters = {
            "especialista_fordka": ["../database/documents/manual-fordka.pdf"],
            "especialista_fiatmobi": ["../database/documents/manual-mobi.pdf"],
            "especialista_generalista": [
                "../database/documents/manual-fordka.pdf", 
                "../database/documents/manual-mobi.pdf"
            ]
        }

        async def busca_documentos(query: str) -> str:
            """Consulta a base de dados dos manuais automotivos."""
            pdfs_para_buscar = pdf_filters.get(agent_name, [])

            if not pdfs_para_buscar:
                return "Nenhum manual configurado para este especialista."

            rag = await initialize_rag()
            
            # Indexa/Carrega os PDFs correspondentes ao agente atual
            for pdf_path in pdfs_para_buscar:
                await index_file(rag, pdf_path)

            resp = await run_async_query(rag, query, "mix")
            return f"[{agent_name}] Resultado da busca:\n{resp}"

        # Retorna a função genérica empacotada como ferramenta
        return [FunctionTool(func=busca_documentos)]
