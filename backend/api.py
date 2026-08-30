from __future__ import annotations

import argparse
import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from starter.agent import Agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_PATH = PROJECT_ROOT / "frontend" / "index.html"
CATALOG_PATH = PROJECT_ROOT / "data" / "catalog.jsonl"
MAX_REQUEST_BYTES = 1_000_000
DEFAULT_PROFILE = {
    "purchase_frequency": "new demo session",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": [],
    "summary": "Interactive video demonstration profile.",
}


def _optional_float(value: object) -> float | None:
    try:
        parsed = float(str(value).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def _category_label(value: object) -> str:
    words = re.findall(r"[A-Za-z0-9&'-]+", str(value or ""))
    return " · ".join(words[-3:]) if words else "Catalog product"


class DemoApplication:
    """Own one warm Agent and adapt its contract for the browser UI."""

    def __init__(self, catalog_path: Path = CATALOG_PATH) -> None:
        self.agent = Agent(catalog_path)
        self.sessions: set[str] = set()

    def close(self) -> None:
        self.agent.close()

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        profile = payload.get("user_profile", DEFAULT_PROFILE)
        if not isinstance(profile, dict):
            raise ValueError("user_profile must be an object")
        self.agent.reset(session_id, profile)
        self.sessions.add(session_id)
        return {"session_id": session_id, "status": "ready"}

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id") or "").strip()
        message = str(payload.get("user_message") or "").strip()
        try:
            turn = int(payload.get("turn"))
            top_k = int(payload.get("top_k", 10))
        except (TypeError, ValueError) as error:
            raise ValueError("turn and top_k must be integers") from error
        if not session_id:
            raise ValueError("session_id is required")
        if not message:
            raise ValueError("user_message is required")
        if not 1 <= turn <= 10:
            raise ValueError("turn must be between 1 and 10")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        if session_id not in self.sessions:
            self.agent.reset(session_id, DEFAULT_PROFILE)
            self.sessions.add(session_id)

        response = self.agent.respond(session_id, message, turn, top_k)
        state = self.agent._sessions[session_id]
        mode = self.agent.search.intent_router.route(state).mode.value
        recommendations = response.get("recommendations") or []
        metadata = self._metadata([
            str(item.get("parent_asin"))
            for item in recommendations
            if isinstance(item, dict) and item.get("parent_asin")
        ])
        enriched: list[dict[str, Any]] = []
        for item in recommendations:
            if not isinstance(item, dict):
                continue
            parent_asin = str(item.get("parent_asin") or "")
            product = metadata.get(parent_asin, {})
            store = str(product.get("store") or "").strip()
            enriched.append({
                "parent_asin": parent_asin,
                "score": float(item.get("score") or 0.0),
                "title": product.get("title") or parent_asin,
                "price": _optional_float(product.get("price")),
                "average_rating": _optional_float(product.get("average_rating")),
                "rating_number": int(float(product.get("rating_number") or 0)),
                "category": _category_label(product.get("categories")),
                "tags": [value for value in (store, f"{mode} route") if value],
            })
        return {
            **response,
            "mode": mode,
            "recommendations": enriched,
        }

    def _metadata(self, parent_asins: list[str]) -> dict[str, dict[str, Any]]:
        unique = list(dict.fromkeys(parent_asins))
        if not unique:
            return {}
        placeholders = ",".join("?" for _ in unique)
        rows = self.agent.search.connection.execute(
            "SELECT parent_asin, title, categories, store, price, "
            "average_rating, rating_number FROM products "
            f"WHERE parent_asin IN ({placeholders})",
            unique,
        ).fetchall()
        keys = (
            "parent_asin", "title", "categories", "store", "price",
            "average_rating", "rating_number",
        )
        return {
            str(row[0]): dict(zip(keys, row))
            for row in rows
        }


class DemoRequestHandler(BaseHTTPRequestHandler):
    app: DemoApplication

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._send_bytes(
                HTTPStatus.OK,
                FRONTEND_PATH.read_bytes(),
                "text/html; charset=utf-8",
            )
            return
        if self.path == "/api/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            if self.path == "/api/reset":
                result = self.app.reset(payload)
            elif self.path == "/api/chat":
                result = self.app.chat(payload)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
        except (json.JSONDecodeError, ValueError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except Exception as error:
            self.log_error("request failed: %s", error)
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "The shopping agent could not complete the request."},
            )
            return
        self._send_json(HTTPStatus.OK, result)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if not 0 < length <= MAX_REQUEST_BYTES:
            raise ValueError("request body is empty or too large")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        self._send_bytes(
            status,
            (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _send_bytes(
        self, status: HTTPStatus, payload: bytes, content_type: str
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the SEAM demonstration UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = DemoApplication()
    DemoRequestHandler.app = app
    server = HTTPServer((args.host, args.port), DemoRequestHandler)
    print(f"SEAM demo available at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
