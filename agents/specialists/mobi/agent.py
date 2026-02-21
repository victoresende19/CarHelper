
# app_agents.py
from __future__ import annotations
from google.adk.tools.agent_tool import AgentTool
from specialists.agent import SpecialistAgent
from tools.query.mobi import query_fiatmobi_database


fiatMobi = AgentTool(agent = SpecialistAgent(
        name="especialista_fiatmobi",
        description=(
            "Especialista técnico no Fiat Mobi. "
            "Use este agente para perguntas sobre especificações, manutenção, "
            "revisões, peças e uso do Fiat Mobi."
            "Considere os documentos recuperados da base de dados do Fiat Mobi para responder às perguntas, sem adicionar conhecimento prévio ou suposições."
            "Utilize as ferramentas de consulta para extrair informações diretamente dos documentos do Fiat Mobi, mesmo que a pergunta pareça simples ou direta. Nunca responda com base em conhecimento prévio ou suposições."
            "Crie a query para as ferramentas de consulta de forma a extrair informações relevantes dos documentos do Fiat Mobi, mesmo que a pergunta pareça simples ou direta. Nunca responda com base em conhecimento prévio ou suposições."
        )
    )
)