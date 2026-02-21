# app_agents.py
from __future__ import annotations
from google.adk.tools.agent_tool import AgentTool
from specialists.agent import SpecialistAgent
from tools.query.fordka import query_fordka_database

fordKa = AgentTool(agent = SpecialistAgent(
        name="especialista_fordka",
        description=(
            "Especialista técnico no Ford KA. "
            "Use este agente para perguntas sobre especificações, manutenção, "
            "revisões, peças e uso do Ford KA."
            "Considere os documentos recuperados da base de dados do Ford KA para responder às perguntas, sem adicionar conhecimento prévio ou suposições."
            "Utilize as ferramentas de consulta para extrair informações diretamente dos documentos do Ford KA, mesmo que a pergunta pareça simples ou direta. Nunca responda com base em conhecimento prévio ou suposições."
            "Crie a query para as ferramentas de consulta de forma a extrair informações relevantes dos documentos do Ford KA, mesmo que a pergunta pareça simples ou direta. Nunca responda com base em conhecimento prévio ou suposições."
        ),
    )
)