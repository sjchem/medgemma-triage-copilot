"""
MedGuard Triage Copilot – UI Launcher
======================================
Convenience entry-point:

  # Full mode (models loaded):
  python launch_ui.py

  # Demo / offline mode (mock backend, no GPU required):
  python launch_ui.py --demo-mode

  # Public share link (Gradio tunnel):
  python launch_ui.py --share

  # Custom host / port:
  python launch_ui.py --host 127.0.0.1 --port 8080
"""

import sys
from pathlib import Path

# Ensure project root is first on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    # Re-use the argparse / launch logic already in gradio_app
    from app.gradio_app import build_app, _mock_result
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="MedGuard Triage Copilot – Gradio UI")
    parser.add_argument("--host",      default="0.0.0.0",  help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",      type=int, default=None, help="Port (default: auto 7860-7880)")
    parser.add_argument("--share",     action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--demo-mode", action="store_true",
                        help="Use mock backend — no models or GPU required")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.demo_mode:
        import app.gradio_app as _gapp
        _gapp._run_text_pipeline       = lambda _t: _mock_result()
        _gapp._run_structured_pipeline = lambda *_a: _mock_result()
        _gapp._DEMO_MODE = True
        print("\n[DEMO MODE] Mock backend active — no models will be loaded.\n")

    from app.gradio_app import _CSS
    ui = build_app()
    # Find a free port in range 7860-7880 if no port explicitly given
    import socket
    port = args.port
    if port is None:
        for candidate in range(7860, 7881):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", candidate)) != 0:
                    port = candidate
                    break
        if port is None:
            port = 7860  # let Gradio raise its own error

    print(f"\n  Starting on http://localhost:{port}\n")
    ui.launch(
        server_name=args.host,
        server_port=port,
        share=args.share,
        css=_CSS,
        inbrowser=True,
    )
