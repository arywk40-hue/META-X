"""Wrapper module exposing the root FastAPI app from the expected server/ path."""

from __future__ import annotations

from app import app, create_app
from server_runner import main as _run_server


def main() -> None:
    _run_server()


if __name__ == "__main__":
    main()


__all__ = ["app", "create_app", "main"]
