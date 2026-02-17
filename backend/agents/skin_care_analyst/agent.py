"""
Skin Care Classifier Agent.
Builds and compiles the LangGraph workflow.

Graph:  classify_node  →  report_node  →  END
"""

import os
import yaml
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from .nodes import classify_node, report_node
from ..global_state import AgentState

load_dotenv()

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


class SkinCareAgent:
    """
    Standard (non-ReAct) agent for skin lesion classification.

    Usage:
        agent  = SkinCareAgent()
        result = agent.run(state)
    """

    def __init__(self, config_path: str = _CONFIG_PATH):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        llm_cfg = self.config["llm"]
        api_key = os.getenv(llm_cfg["api_key_env"])
        if not api_key:
            raise ValueError(f"Missing env var: {llm_cfg['api_key_env']}")

        # Plain LLM — no tools bound (tool is called directly in classify_node)
        self.llm = ChatOpenAI(
            model=llm_cfg["model"],
            temperature=llm_cfg.get("temperature", 1),
            openai_api_key=api_key,
            base_url=llm_cfg.get("base_url"),
            max_tokens=llm_cfg.get("max_tokens"),
        )

        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("classify", lambda s: classify_node(s, self.llm))
        workflow.add_node("report",   lambda s: report_node(s, self.llm))

        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "report")
        workflow.add_edge("report", END)

        return workflow.compile()

    def run(self, state: AgentState) -> AgentState:
        """Run the agent and return the updated state."""
        print("\n" + "=" * 55)
        print("🩺 SKIN CARE AGENT: Starting Classification")
        print("=" * 55)

        final_state = self.graph.invoke(state)

        print("\n✅ Skin Care Agent complete.")
        print(f"   Label  : {final_state.get('vision_results', {}).get('label', 'N/A')}")
        print(f"   Report : {str(final_state.get('final_report', ''))[:80]}...")
        return final_state
