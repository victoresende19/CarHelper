# app_agents.py
from __future__ import annotations

from google.adk.tools.function_tool import FunctionTool
from typing import AsyncGenerator, List

from google.adk.agents import Agent, ParallelAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.apps.app import App
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.tools.base_tool import BaseTool

from recepcionist.agent import ReceptionistAgent
from didactic.agent import DidacticAgent
from security.agent import SecurityAgent
from tom.agent import FinalAgent

from specialists.fordka.agent import fordKa
from specialists.mobi.agent import fiatMobi
from specialists.generalist.agent import generalista
from tools.flows.done import mark_flow_done


class CarHelperMasterAgent(BaseAgent):
    receptionist: Agent
    parallel_review: ParallelAgent
    finalizer: Agent

    def __init__(
            self,
            tools: List[BaseTool]
    ):
        receptionist    = ReceptionistAgent(tools=tools)
        security        = SecurityAgent()
        didactic        = DidacticAgent()
        finalizer       = FinalAgent()
        parallel_review = ParallelAgent(
            name="agentes_paralelos_validacao",
            sub_agents=[security, didactic]
        )

        super().__init__(
            name="fluxo_carros_lightrag_openai_master",
            receptionist=receptionist,
            parallel_review=parallel_review,
            finalizer=finalizer,
            sub_agents=[
                receptionist,
                parallel_review,
                finalizer,
            ],
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        ctx.session.state["flow_done"] = False
        ctx.session.state["quality_done"] = False

        # Inicializa variáveis para evitar KeyError nos agentes de revisão
        ctx.session.state.setdefault("resposta_consultor", "")
        ctx.session.state.setdefault("revisao_seguranca", "STATUS: SEGURO\nMOTIVO: Nenhuma resposta técnica gerada.")
        ctx.session.state.setdefault("resposta_rascunho", "")

        # 1) Recepcionista
        async for event in self.receptionist.run_async(ctx):
            yield event

        # Early return FORA do async for — funciona corretamente
        if ctx.session.state.get("flow_done") is True:
            return

        for _ in range(5):
            # 3) Loop: paralelo segurança + didática
            async for event in self.parallel_review.run_async(ctx):
                yield event

            if ctx.session.state.get("quality_done") is True:
                break

        # 4) Finalizador
        async for event in self.finalizer.run_async(ctx):
            yield event


# =========================
# 7) App with cache
# =========================
tools = [
    FunctionTool(func=mark_flow_done),
    # FunctionTool(func=mark_quality_done),
    fordKa,
    fiatMobi,
    generalista,
]
root_agent = CarHelperMasterAgent(tools=tools)

app = App(
    name="master",
    root_agent=root_agent,
    context_cache_config=ContextCacheConfig(
        min_tokens=2048,
        ttl_seconds=600,
        cache_intervals=5,
    ),
)
