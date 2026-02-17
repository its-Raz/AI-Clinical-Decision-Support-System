from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage
import json

def _print_messages(messages):
    """Print messages in a clean, readable format."""
    print("\n" + "─" * 60)
    print("📨 MESSAGES SENT TO LLM:")
    print("─" * 60)

    for i, msg in enumerate(messages, 1):
        if isinstance(msg, SystemMessage):
            print(f"\n[{i}] 🔧 SYSTEM:")
            # Show first 150 chars
            content = msg.content + "..." if len(msg.content) > 150 else msg.content
            print(f"    {content}")

        elif isinstance(msg, HumanMessage):
            print(f"\n[{i}] 👤 HUMAN:")
            # Show first 150 chars
            content = msg.content + "..." if len(msg.content) > 150 else msg.content
            print(f"    {content}")

        elif isinstance(msg, AIMessage):
            print(f"\n[{i}] 🤖 AI:")
            if msg.content:
                content = msg.content + "..." if len(msg.content) > 150 else msg.content
                print(f"    Content: {content}")
            if msg.tool_calls:
                print(f"    Tool Calls: {[tc['name'] for tc in msg.tool_calls]}")

        elif isinstance(msg, ToolMessage):
            print(f"\n[{i}] 🔨 TOOL ({msg.name}):")
            # Show first 100 chars of result
            content = msg.content + "..." if len(msg.content) > 100 else msg.content
            print(f"    {content}")

    print("\n" + "─" * 60)


def _print_response(response):
    """Print LLM response in a clean, readable format."""
    print("\n" + "─" * 60)
    print("🎯 LLM RESPONSE:")
    print("─" * 60)

    # Content
    print(f"\n💬 Content:")
    if response.content:
        # Print full content if short, or first 300 chars
        if len(response.content) <= 300:
            print(f"   {response.content}")
        else:
            print(f"   {response.content}...")
            print(f"   ... ({len(response.content)} total chars)")
    else:
        print(f"   (empty)")

    # Tool calls
    print(f"\n🔧 Tool Calls:")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for i, tc in enumerate(response.tool_calls, 1):
            print(f"   {i}. {tc['name']}")
            print(f"      Args: {json.dumps(tc['args'], indent=13)[1:]}")  # Indent properly
    else:
        print(f"   (none)")

    # Metadata
    metadata = response.response_metadata
    token_usage = metadata.get('token_usage', {})

    print(f"\n📊 Metadata:")
    print(f"   Model: {metadata.get('model_name', 'N/A')}")
    print(f"   Finish Reason: {metadata.get('finish_reason', 'N/A')}")
    print(f"   Tokens: {token_usage.get('total_tokens', 'N/A')} total")
    print(f"           ├─ Prompt: {token_usage.get('prompt_tokens', 'N/A')}")
    print(f"           ├─ Completion: {token_usage.get('completion_tokens', 'N/A')}")

    # Show reasoning tokens if present
    details = token_usage.get('completion_tokens_details', {})
    if details.get('reasoning_tokens', 0) > 0:
        print(f"           └─ Reasoning: {details['reasoning_tokens']} (internal thinking)")

    print("─" * 60 + "\n")