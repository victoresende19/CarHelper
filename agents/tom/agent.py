from __future__ import annotations
from google.adk.agents import LlmAgent
from agents.tools.config import configure_model


class FinalAgent(LlmAgent):
    def __init__(self):
        super().__init__(
            name="agente_final",
            model=configure_model(),
            description="Entrega a resposta final ao usuário com tom amigável.",
            instruction=(
                "Retorne ao usuário apenas o texto final pronto, com base em: {resposta_rascunho}\n"
                "Escreva em tom amigável e acolhedor, em português do Brasil.\n"
                "Não inclua logs, notas internas ou campos estruturados."
            ),
            output_key="resposta_final",
        )