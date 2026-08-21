"""Test helper: run the FastAPI app on a real socket in a background thread.

The API test-suite drives servers with urllib over HTTP, so an in-process
ASGI test client is not enough; this wrapper gives the FastAPI app the same
lifecycle surface (``base_url`` + ``shutdown()``) as a ThreadingHTTPServer.
"""

from __future__ import annotations

import threading
import time

import uvicorn


class ThreadedApiServer:
    def __init__(self, app, *, host: str = "127.0.0.1", port: int = 0):
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 15
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("API server thread exited during startup")
            if time.monotonic() > deadline:
                raise RuntimeError("API server did not start within 15s")
            time.sleep(0.01)
        sock = self._server.servers[0].sockets[0]
        self.server_address = sock.getsockname()[:2]

    @property
    def base_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}"

    def shutdown(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=15)
