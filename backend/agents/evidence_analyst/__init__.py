"""
ReAct Agent - Class-based implementation with Google's TypedDict pattern.
"""

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
import yaml

from backend.agents.evidence_analyst.state import ReActInternalState
from backend.agents.evidence_analyst.nodes import call_model, call_tool
from backend.agents.evidence_analyst.edges import should_continue

# Import tools from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from backend.agents.evidence_analyst.tools import search_medical_knowledge

load_dotenv()


class ReActAgent:
    """
    ReAct Agent for clinical information gathering.

    Combines:
    - Google's TypedDict state pattern
    - Class-based encapsulation for reusability

    Usage:
        agent = ReActAgent()
        result = agent.run(global_state)
    """

    def __init__(self, config_path: str = None):
        """
        Initialize ReAct agent.

        Args:
            config_path: Path to config.yaml (optional, defaults to blood_test_analyst/config.yaml)
        """
        # Load config from blood_test_analyst folder (parent directory)
        if config_path is None:
            # Go up one level from react_agent to blood_test_analyst folder
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Get API key
        api_key = os.getenv(self.config['llm']['api_key_env'])
        if not api_key:
            raise ValueError(f"Missing {self.config['llm']['api_key_env']} in .env")

        # Create LLM with all config parameters
        llm_config = self.config['llm']
        self.llm = ChatOpenAI(
            model=llm_config['model'],
            temperature=llm_config['temperature'],
            openai_api_key=api_key,
            base_url=llm_config.get('base_url'),  # Optional base_url
            max_tokens=llm_config.get('max_tokens'),  # Optional max_tokens
            reasoning_effort="low"
        )

        # Define tools
        self.tools = [

            search_medical_knowledge
        ]

        # Bind tools to model
        self.model = self.llm.bind_tools(self.tools)

        # Create tools lookup dictionary
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        # Get max iterations
        self.max_iterations = self.config['react']['max_iterations']

        # Build the graph
        self.graph = self._build_graph()


    def _build_graph(self):
        """
        Build the LangGraph workflow.

        Returns:
            Compiled graph
        """
        workflow = StateGraph(ReActInternalState)

        # Add nodes (bind self to each node function)
        workflow.add_node("llm", lambda state: call_model(self, state))
        workflow.add_node("tools", lambda state: call_tool(self, state))

        # Set entry point
        workflow.set_entry_point("llm")

        # Add conditional edge from llm
        workflow.add_conditional_edges(
            "llm",
            lambda state: should_continue(self, state),
            {
                "continue": "tools",
                "end": END,
            },
        )

        # Add edge from tools back to llm
        workflow.add_edge("tools", "llm")

        # Compile and return
        return workflow.compile()


# Export
__all__ = ['ReActAgent']