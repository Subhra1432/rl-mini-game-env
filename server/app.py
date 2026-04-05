"""
FastAPI application for the Email Triage Environment.

This module creates an HTTP server that exposes the EmailTriageEnvironment
over HTTP and WebSocket endpoints, compatible with MCPToolClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .email_triage_environment import EmailTriageEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.email_triage_environment import EmailTriageEnvironment

# Create the app with web interface support
# Pass the class (factory) for WebSocket session support
app = create_app(
    EmailTriageEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="email_triage_env",
)


def main():
    """
    Entry point for direct execution.

    Enables running the server without Docker:
        python -m server.app
        uv run --project . server
    """
    import uvicorn

    port = int(__import__("os").environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
