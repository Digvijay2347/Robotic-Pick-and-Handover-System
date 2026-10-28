"""
Turnkey Launcher for the Dual-Arm Robotic Pick-and-Handover Web Platform.

Starts the Uvicorn ASGI server hosting the FastAPI application and optionally
opens the web dashboard in the default browser.

Usage:
    python launch_platform.py [--host 0.0.0.0] [--port 8000] [--no-browser]
"""

import os
import sys
import argparse
import webbrowser
import threading
import time

# Ensure project root in sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import uvicorn


def open_browser_delayed(url: str, delay_sec: float = 1.5):
    """Waits for server initialization before launching browser tab."""
    time.sleep(delay_sec)
    print(f"\n[Platform Launcher] Opening browser dashboard at: {url}")
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="Dual-Arm Robotic Web Platform Launcher")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="HTTP/WebSocket port")
    parser.add_argument("--no-browser", action="store_true", help="Do not launch browser window automatically")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print("=" * 80)
    print("      LAUNCHING BIMANUAL UR10e ROBOTIC WEB PLATFORM (PHASE 1)")
    print("=" * 80)
    print(f"  Access URL        : {url}")
    print(f"  API Docs (Swagger): {url}/docs")
    print(f"  WebSocket Endpoint: ws://{args.host}:{args.port}/ws/telemetry")
    print(f"  Video Stream      : {url}/api/stream/feed/overhead_cam")
    print("=" * 80 + "\n")

    if not args.no_browser:
        threading.Thread(target=open_browser_delayed, args=(url,), daemon=True).start()

    uvicorn.run("web.server.main:app", host=args.host, port=args.port, log_level="info", reload=False)


if __name__ == "__main__":
    main()
