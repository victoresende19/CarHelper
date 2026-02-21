from __future__ import annotations
from typing import List
from google.adk.tools.base_tool import BaseTool
from google.adk.agents import LlmAgent
from tools.config import configure_model

class ReceptionistAgent(LlmAgent):
    tools: List[BaseTool]

    def __init__(
            self,
            tools: List[BaseTool],
    ):
        super().__init__(
            name="agente_recepcionista",
            model=configure_model(),
            description="Recepciona o usuário e explica escopo do sistema.",
            instruction=(
                "Você é o recepcionista do sistema de dúvidas sobre Ford KA e Fiat Mobi.\n\n"
                "Analise a mensagem do usuário e siga UMA das duas regras abaixo:\n\n"
                "REGRA A — Mensagem SEM pergunta técnica (apenas cumprimento, 'oi', 'olá', 'tudo bem', etc.):\n"
                "  - Responda com uma saudação amigável.\n"
                "  - Explique em 1-2 frases que o sistema responde sobre informações técnicas,\n"
                "    comparações, manutenção e uso do Ford KA e Fiat Mobi.\n"
                "  - Convide o usuário a fazer uma pergunta.\n"
                "  - Chame a tool 'mark_flow_done'.\n\n"
                "REGRA B - perguntas fora do escopo de carros:\n"
                "  - Informe educadamente que o sistema é focado em Ford KA e Fiat Mobi.\n"
                "  - chame 'mark_flow_done'.\n"
                "REGRA C - Para perguntas técnicas, identifique a ferramenta mais relevante para responder à pergunta técnica e chame a ferramenta correspondente, passando a pergunta do usuário como argumento.\n"
                "  - Use 'especialista_fordka' para perguntas claramente relacionadas ao Ford KA.\n"
                "  - Use 'especialista_fiatmobi' para perguntas claramente relacionadas ao Fiat Mobi.\n"
                "  - Use 'especialista_generalista' para perguntas técnicas mais genéricas sobre carros, trazendo informações do Ford KA e Fiat Mobi.\n"
            ),
            tools=tools,
            output_key="resposta_recepcionista",
        )