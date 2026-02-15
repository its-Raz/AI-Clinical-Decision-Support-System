# Purpose: The Manager agent logic
# Contains:
#
# manager_node() function - the main node function
# Routing decision logic (which agent to call next)
# State update logic (sets next_action field)
# LLM prompt construction for routing decisions
#
# What it does:
#
# Receives full state
# Analyzes what's been done (has diagnosis? patient contacted? urgent?)
# Decides: route to Analyst, Investigator, Secretary, or END
# Returns: {"next_action": "analyst"} or similar