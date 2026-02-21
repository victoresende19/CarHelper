from tools.config import configure_model
from google.adk.tools.base_tool import BaseTool
from google.adk.agents import LlmAgent
from typing import List
from tools.toolset import ManualToolset

class SpecialistAgent(LlmAgent):
    def __init__(
            self,
            name: str,
            description: str
    ):
        super().__init__(
            name=name,
            model=configure_model(),
            description=description,
            instruction=(
                "Fornecer respostas precisas a perguntas e solicitações baseando-se exclusivamente em informações extraídas de documentos recupeerados, sem conhecimento prévio ou suposições. Siga rigorosamente as seguintes regras:\n\n"
                "1) Analise a pergunta do usuário e identifique as principais pontos da dúvida ou solicitação\n"
                "2) Sempre realize buscas nas ferramentas disponibilizadas para encontrar informações relevantes, mesmo que a pergunta pareça simples ou direta. Nunca responda com base em conhecimento prévio ou suposições.\n" \
                "3) Leia atentamente os documentos recuperados e seleciona os trechos mais relevantes para responder à pergunta do usuário. **NÃO ADICIONE CONHECIMENTO PRÉVIO**\n"
            ),
            tools=[ManualToolset()],
            output_key="resposta_consultor",
        )
