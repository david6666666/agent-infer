# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project

"""Loopback-only demo API; no serving engine, shell, or deployment adapters."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from agentinfer.rsi.controller import Controller
from agentinfer.rsi.feedback import list_layers
from agentinfer.rsi.knowledge import KnowledgeRepository


def make_handler(db_path, run_id):
    """Create a handler for a single demonstration run, without starting a server."""
    db_path = Path(db_path)

    class Handler(BaseHTTPRequestHandler):
        def _allowed_host(self):
            port = self.server.server_address[1]
            return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def _send(self, value, status=200, content_type="application/json; charset=utf-8"):
            body = (
                value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._allowed_host():
                self._send({"error": "Loopback Host header required"}, 403)
                return
            request = urlsplit(self.path)
            query = parse_qs(request.query)
            try:
                if request.path in ("/", "/dashboard.html"):
                    html = files("agentinfer.rsi.dashboard").joinpath("static/dashboard.html").read_bytes()
                    self._send(html, content_type="text/html; charset=utf-8")
                elif request.path == "/api/rsi/snapshot":
                    controller = Controller(db_path)
                    self._send({"synthetic": True, "production_connected": False, **controller.snapshot(run_id)})
                elif request.path == "/api/rsi/taxonomy":
                    self._send(list_layers(query.get("backend", ["cuda"])[0]))
                elif request.path == "/api/rsi/knowledge":
                    required = ("scope", "backend", "component_version", "workload")
                    if any(key not in query for key in required):
                        raise ValueError("Provide scope, backend, component_version, and workload")
                    self._send(
                        KnowledgeRepository(db_path).search(
                            **{key: query[key][0] for key in required}, text=query.get("text", [""])[0]
                        )
                    )
                else:
                    self._send({"error": "Unknown endpoint"}, 404)
            except (KeyError, ValueError, RuntimeError) as error:
                self._send({"error": str(error)}, 400)

        def do_POST(self):
            # Consume a bounded body before replying: closing an unread POST can
            # reset the connection on Windows and discard the rejection response.
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 65536:
                    raise ValueError("Command body must be between 1 and 65536 bytes")
                self.connection.settimeout(5)
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError("Incomplete command body")
            except (ValueError, TimeoutError) as error:
                self._send({"error": str(error)}, 400)
                return
            if not self._allowed_host():
                self._send({"error": "Loopback Host header required"}, 403)
                return
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self._send({"error": "Cross-origin commands are not accepted"}, 403)
                return
            if urlsplit(self.path).path != "/api/rsi/commands":
                self._send({"error": "Unknown endpoint"}, 404)
                return
            if self.headers.get_content_type() != "application/json":
                self._send({"error": "application/json required"}, 415)
                return
            try:
                command = json.loads(body)
                if not isinstance(command, dict):
                    raise ValueError("Command must be an object")
                controller = Controller(db_path)
                if not controller.get(run_id)["demo"]:
                    self._send({"error": "HTTP commands are restricted to demo runs"}, 403)
                    return
                result = controller.command(
                    run_id,
                    expected_revision=command["expected_revision"],
                    idempotency_key=command["idempotency_key"],
                    action=command["action"],
                    payload=command.get("payload", {}),
                )
                self._send({"synthetic": True, "run": result, "production_connected": False})
            except (KeyError, ValueError, RuntimeError) as error:
                self._send({"error": str(error)}, 409)

    return Handler


def serve(run_dir, *, run_id="demo-r024", port=8877):
    """Serve only on IPv4 loopback. Caller must initialize a demo run first."""
    db_path = Path(run_dir) / "state.sqlite"
    if not Controller(db_path).get(run_id)["demo"]:
        raise ValueError("The bootstrap HTTP server only serves demo runs")
    with ThreadingHTTPServer(("127.0.0.1", port), make_handler(db_path, run_id)) as server:
        print(f"RSI demo: http://127.0.0.1:{server.server_address[1]}/dashboard.html", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
