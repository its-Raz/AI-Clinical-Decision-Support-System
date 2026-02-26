"""
Edge / routing logic for the Manager graph.
"""


def route_after_manager(state: dict) -> str:
    """
    Called after manager_node.
    Returns the name of the next graph node.
    """
    next_step = state.get("next_step", "unknown")

    routes = {
        "blood_test_analyst": "blood_test_analyst",
        "skin_care_analyst":  "skin_care_analyst",
        "evidence_analyst":   "evidence_analyst",
        "clarification_needed": "deliver",
        "unsupported":          "deliver",
    }

    destination = routes.get(next_step, "deliver")
    print(f"   ↳ Edge resolved: {destination}")
    return destination