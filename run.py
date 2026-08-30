#!/usr/bin/env python3
"""
Minimal launcher for the ProtocolsNerd backend + frontend file server.
"""

import subprocess
import sys
import webbrowser
import time
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent / "protocolnerd-backend"
FRONTEND_DIR = Path(__file__).resolve().parent / "protocolnerd-website"
VENV_PYTHON = Path(__file__).resolve().parent / "venv" / "bin" / "python"
PYTHON_EXE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
BACKEND_PORT = 8001
FRONTEND_PORT = 5555


def check_ollama():
    """Quick check whether Ollama appears reachable."""
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2):
            return True
    except Exception:
        return False


def main():
    if not check_ollama():
        print("\n[WARNING] Ollama does not appear to be running on localhost:11434.")
        print("  Start it with:  ollama serve")
        print("  Pull a model:   ollama pull llama3.2\n")

    # Start the FastAPI backend
    backend_proc = subprocess.Popen(
        [PYTHON_EXE, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(BACKEND_PORT)],
        cwd=str(BACKEND_DIR),
    )

    # Start a simple static file server for the frontend
    frontend_proc = subprocess.Popen(
        [PYTHON_EXE, "-m", "http.server", str(FRONTEND_PORT)],
        cwd=str(FRONTEND_DIR),
    )

    time.sleep(2)
    url = f"http://localhost:{FRONTEND_PORT}/chat.html"
    print(f"\n  Backend:  http://localhost:{BACKEND_PORT}")
    print(f"  Chat UI:  {url}\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend_proc.terminate()
        frontend_proc.terminate()


if __name__ == "__main__":
    main()
