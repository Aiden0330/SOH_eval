"""FastAPI surface for sohbench (drop-in replacement for sohbench.api_server)."""

from .app import create_app, main, serve

__all__ = ["create_app", "main", "serve"]
