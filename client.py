"""
Email Triage Environment Client.

Provides the client for connecting to an Email Triage Environment server.
Extends MCPToolClient for tool-calling style interactions.

Example:
    >>> with EmailTriageEnv(base_url="http://localhost:8000") as env:
    ...     env.reset(task_id="email_classify")
    ...
    ...     # Discover tools
    ...     tools = env.list_tools()
    ...     print([t.name for t in tools])
    ...     # ['classify_email', 'route_email', 'draft_response', 'get_email_details']
    ...
    ...     # Classify the email
    ...     result = env.call_tool("classify_email",
    ...         category="technical", priority="high")
    ...     print(result)

Example with sync wrapper:
    >>> with EmailTriageEnv(base_url="http://localhost:8000").sync() as env:
    ...     env.reset(task_id="email_triage")
    ...     result = env.call_tool("classify_email",
    ...         category="billing", priority="high")
    ...     result = env.call_tool("route_email", department="billing")
"""

from openenv.core.mcp_client import MCPToolClient


class EmailTriageEnv(MCPToolClient):
    """
    Client for the Email Triage Environment.

    Inherits all functionality from MCPToolClient:
    - list_tools(): Discover available tools
    - call_tool(name, **kwargs): Call a tool by name
    - reset(**kwargs): Reset the environment
    - step(action): Execute an action
    - state(): Get current state

    Available tools:
    - classify_email(category, priority): Classify the email
    - route_email(department): Route to the correct department
    - draft_response(response_text): Draft a response
    - get_email_details(): Get additional email details
    """

    pass  # MCPToolClient provides all needed functionality
