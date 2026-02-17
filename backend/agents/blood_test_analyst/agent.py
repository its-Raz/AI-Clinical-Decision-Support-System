# Purpose: Main Analyst agent orchestration
# Contains:
#
# analyst_node() function - the main node function called by the graph
# High-level orchestration: ReAct → Draft → Reflexion loop
# Calls sub-nodes (react.py, draft.py, critique.py, revise.py)
# Final output assembly
#
# What it does:
# Function: analyst_node(state)
#
# Print
# "Starting Analyst..."
#
# # Phase 1: Gather info
# state = react_node(state)
#
# # Phase 2: Create quality diagnosis
# state = reflexion_node(state)
#
# # Return all updates
# return state