from __future__ import annotations
from typing import List
from google.adk.tools.base_tool import BaseTool
from google.adk.agents import LlmAgent
from agents.tools.config import configure_model
from agents.tools.flows.quality import mark_quality_done

class DidacticAgent(LlmAgent):
    """
    Roda em paralelo com SecurityAgent.
    O KeyError é evitado inicializando 'revisao_seguranca' no state antes do loop.
    Após o paralelo, lê o valor já populado pelo SecurityAgent da mesma rodada.
    """

    def __init__(
            self,
            tools: List[BaseTool] = [mark_quality_done],
    ):
        super().__init__(
            name="agente_didatica",
            model=configure_model(),
            description="Avalia clareza, refina se necessário e sinaliza fim do loop de qualidade.",
            tools=tools,
            instruction=(
                "Você avalia e, se necessário, melhora a resposta atual.\n\n"
                "Entradas disponíveis:\n"
                "- Resposta atual: {resposta_consultor}\n"
                "- Resultado de segurança: {revisao_seguranca}\n\n"
                "Siga EXATAMENTE estas regras em ordem:\n\n"
                "1) Se 'revisao_seguranca' contiver STATUS: BLOQUEADO:\n"
                "   - Substitua o rascunho por: 'Desculpe, não posso responder a essa solicitação.'\n"
                "   - Chame a tool 'mark_quality_done'.\n"
                "   - Retorne apenas o texto de bloqueio.\n\n"
                "2) Avalie a clareza e didática do rascunho. Dê uma nota interna de 0 a 5.\n"
                "   - Nota >= 4 → APROVADO:\n"
                "     * Não altere o conteúdo.\n"
                "     * Chame a tool 'mark_quality_done'.\n"
                "     * Retorne o rascunho como está.\n"
                "   - Nota < 4 → REVISAR:\n"
                "     * Reescreva a resposta de forma mais clara, objetiva e didática.\n"
                "     * NÃO chame 'mark_quality_done'.\n"
                "     * Retorne apenas o texto reescrito (sem notas, sem explicações extras).\n"
            ),
            output_key="resposta_rascunho",
        )