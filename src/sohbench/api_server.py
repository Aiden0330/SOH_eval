from __future__ import annotations

import argparse
import base64
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .evaluation_service import EvaluationService
from .model_store import ModelStoreError
from .trusted_plugins import TrustedPluginError


def create_handler(service: EvaluationService):
    class SohbenchApiHandler(BaseHTTPRequestHandler):
        server_version = "sohbench-api/0.1"

        def do_OPTIONS(self):  # noqa: N802
            self._send_json({"ok": True})

        def do_GET(self):  # noqa: N802
            try:
                path = urlparse(self.path).path
                if path == "/api/health":
                    self._send_json({"ok": True, "mode": "trusted_local"})
                elif path == "/api/catalog":
                    self._send_json(service.catalog())
                elif path == "/api/models":
                    self._send_json({"models": service.model_store.catalog()})
                elif path == "/api/jobs":
                    self._send_json({"jobs": service.list_jobs()})
                elif path == "/api/inference-jobs":
                    self._send_json({"jobs": service.list_inference_jobs()})
                elif path.startswith("/api/inference-jobs/"):
                    job_id = path.rsplit("/", 1)[-1]
                    self._send_json(service.get_inference_job(job_id))
                elif path.startswith("/api/jobs/"):
                    job_id = path.rsplit("/", 1)[-1]
                    self._send_json(service.get_job(job_id))
                else:
                    self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            except KeyError:
                self._send_json({"error": "job not found"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self):  # noqa: N802
            try:
                path = urlparse(self.path).path
                payload = self._read_json()
                if path == "/api/upload/dataset":
                    resource = service.store.upload_dataset(
                        payload.get("name"),
                        payload.get("filename", ""),
                        _decode_content(payload),
                        replace=bool(payload.get("replace", False)),
                    )
                    self._send_json({"ok": True, "resource": resource})
                elif path == "/api/upload/algorithm":
                    resource = service.store.upload_estimator_plugin(
                        payload.get("name"),
                        payload.get("filename", ""),
                        _decode_content(payload),
                        replace=bool(payload.get("replace", False)),
                    )
                    self._send_json({"ok": True, "resource": resource})
                elif path == "/api/upload/featureset":
                    resource = service.store.upload_featureset_plugin(
                        payload.get("name"),
                        payload.get("filename", ""),
                        _decode_content(payload),
                        replace=bool(payload.get("replace", False)),
                    )
                    self._send_json({"ok": True, "resource": resource})
                elif path == "/api/upload/model":
                    resource = service.model_store.upload_model(
                        payload.get("name"),
                        payload.get("modelFilename", ""),
                        _decode_named_content(payload, "modelContentBase64"),
                        payload.get("manifestFilename", ""),
                        _decode_named_content(payload, "manifestContentBase64"),
                        replace=bool(payload.get("replace", False)),
                    )
                    self._send_json({"ok": True, "resource": resource})
                elif path == "/api/jobs":
                    self._send_json(service.create_job(payload, start=True), status=HTTPStatus.ACCEPTED)
                elif path == "/api/inference-jobs":
                    self._send_json(service.create_inference_job(payload, start=True), status=HTTPStatus.ACCEPTED)
                else:
                    self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            except (TrustedPluginError, ModelStoreError, ValueError, KeyError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, fmt, *args):
            return

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            return json.loads(body.decode("utf-8"))

        def _send_json(self, payload: dict, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(data)

    return SohbenchApiHandler


def _decode_content(payload: dict) -> bytes:
    encoded = payload.get("contentBase64")
    if not encoded:
        raise ValueError("contentBase64 is required")
    return base64.b64decode(encoded)


def _decode_named_content(payload: dict, key: str) -> bytes:
    encoded = payload.get(key)
    if not encoded:
        raise ValueError(f"{key} is required")
    return base64.b64decode(encoded)


def serve(
    *,
    host: str,
    port: int,
    data_dir: str | Path,
    work_dir: str | Path,
    runs_dir: str | Path,
    frontend_data_dir: str | Path,
) -> None:
    service = EvaluationService(
        data_dir=data_dir,
        work_dir=work_dir,
        runs_dir=runs_dir,
        frontend_data_dir=frontend_data_dir,
    )
    httpd = ThreadingHTTPServer((host, port), create_handler(service))
    print(f"sohbench API listening on http://{host}:{port}", flush=True)
    print(f"data_dir={Path(data_dir)}", flush=True)
    httpd.serve_forever()


def parse_args(argv=None):
    parser = argparse.ArgumentParser("sohbench-api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--data-dir", default=r"D:\battery_SOH\data")
    parser.add_argument("--work-dir", default="work/trusted_plugins")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--frontend-data-dir", default="frontend/app/data")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    serve(
        host=args.host,
        port=args.port,
        data_dir=args.data_dir,
        work_dir=args.work_dir,
        runs_dir=args.runs_dir,
        frontend_data_dir=args.frontend_data_dir,
    )


if __name__ == "__main__":
    main()
