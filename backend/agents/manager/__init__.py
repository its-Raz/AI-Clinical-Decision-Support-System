"""
Manager Agent — full LangGraph orchestration graph.

Graph topology:
  [manager] ──routes──► [blood_test_analyst] ──► [deliver] ──► END
                       ↗ (skin_care reserved for next iteration)
"""

import os
import logging
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .nodes import manager_node, deliver_node
from .edges import route_after_manager

load_dotenv()

log = logging.getLogger(__name__)

# Enable basic logging so debug prints appear in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


class ManagerAgent:
    def __init__(self, config_path: str = _CONFIG_PATH):
        print("\n🏗️  [ManagerAgent] Initialising …")
        log.info("ManagerAgent.__init__: config_path=%s", config_path)

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        cfg     = self.config["llm"]
        api_key = os.getenv(cfg["api_key_env"])

        if not api_key:
            log.error("Missing API key env var: %s", cfg["api_key_env"])
            raise ValueError(f"Missing env var: {cfg['api_key_env']}")

        print(f"   model    : {cfg['model']}")
        print(f"   temp     : {cfg.get('temperature', 0.2)}")
        log.info("ManagerAgent: LLM model=%s", cfg["model"])

        self.llm = ChatOpenAI(
            model=cfg["model"],
            temperature=cfg.get("temperature", 0.2),
            openai_api_key=api_key,
            base_url=cfg.get("base_url"),
            max_tokens=cfg.get("max_tokens"),
        )

        print("   ✅ LLM ready")
        self.graph = self._build_graph()
        print("   ✅ Graph compiled\n")

    def _build_graph(self):
        print("🔧 [ManagerAgent] Building graph …")
        log.info("ManagerAgent._build_graph: start")

        # Deferred imports to avoid circular references and heavy ML library loading
        # Import the node functions directly, not the modules that contain them,
        # to avoid triggering model loads at graph-build time
        from backend.agents.global_state import AgentState

        log.debug("_build_graph: imports resolved (AgentState)")
        print("   ✅ Imports resolved (AgentState)")
        print("   ⚡ Heavy ML imports deferred until node execution")

        # Import node functions with lazy wrappers
        def lazy_blood_test_analyst(state):
            from backend.agents.blood_test_analyst.run_batch import run_batch_analyst
            return run_batch_analyst(state)

        def lazy_skin_care_analyst(state):
            from backend.agents.skin_care_analyst.run import run_skin_care_analyst
            return run_skin_care_analyst(state)

        workflow = StateGraph(AgentState)

        workflow.add_node("manager",            lambda s: manager_node(s, self.llm))
        workflow.add_node("blood_test_analyst",  lazy_blood_test_analyst)
        workflow.add_node("skin_care_analyst",   lazy_skin_care_analyst)
        workflow.add_node("deliver",            lambda s: deliver_node(s, self.llm))

        workflow.set_entry_point("manager")

        workflow.add_conditional_edges(
            "manager",
            route_after_manager,
            {
                "blood_test_analyst": "blood_test_analyst",
                "skin_care_analyst":  "skin_care_analyst",
                "deliver":            "deliver",
            },
        )
        workflow.add_edge("blood_test_analyst", "deliver")
        workflow.add_edge("skin_care_analyst",  "deliver")
        workflow.add_edge("deliver", END)

        log.info("_build_graph: nodes=%s", ["manager", "blood_test_analyst", "skin_care_analyst", "deliver"])
        return workflow.compile()

    def run(self, initial_state: dict) -> dict:
        print("\n" + "=" * 60)
        print("🏥  CLINICAL SYSTEM: Pipeline starting")
        print(f"    patient_id   : {initial_state.get('patient_id')}")
        print(f"    request_type : {initial_state.get('request_type')}")
        print(f"    batch size   : {len(initial_state.get('lab_result') or [])} metrics")
        print("=" * 60)

        log.info(
            "ManagerAgent.run: patient=%s request=%s",
            initial_state.get("patient_id"),
            initial_state.get("request_type"),
        )

        try:
            result = self.graph.invoke(initial_state)
            log.info("ManagerAgent.run: complete. final_report=%d chars",
                     len(result.get("final_report") or ""))
            print("\n✅ Pipeline complete.")
            return result

        except Exception as e:
            log.error("ManagerAgent.run: FAILED — %s", e, exc_info=True)
            print(f"\n❌ Pipeline error: {e}")
            raise


__all__ = ["ManagerAgent"]