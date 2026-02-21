# app_agents.py
from __future__ import annotations
from google.adk.tools.agent_tool import AgentTool
from agents.specialists.agent import SpecialistAgent

generalista = AgentTool(agent = SpecialistAgent(
        name="especialista_generalista",
        description=(
            "Especialista técnico generalista em carros. "
            "Use este agente para perguntas técnicas mais genéricas sobre carros, "
            "pressão de pneu, revisão, manutenção, comparação, etc., quando não for claro se a dúvida é sobre Ford KA ou Fiat Mobi."
            "Este agente trará informações de ambos os carros para fornecer uma resposta mais completa."
            "Considere os documentos recuperados das bases de dados do Ford KA e Fiat Mobi para responder às perguntas, sem adicionar conhecimento prévio ou suposições."
            "Utilize as ferramentas de consulta para extrair informações diretamente dos documentos do Ford KA e Fiat Mobi, mesmo que a pergunta pareça simples ou direta. Nunca responda com base em conhecimento prévio ou suposições."
            "Crie a query para as ferramentas de consulta de forma a extrair informações relevantes dos documentos do Ford KA e Fiat Mobi, mesmo que a pergunta pareça simples ou direta. Nunca responda com base em conhecimento prévio ou suposições."
        )
    )
)