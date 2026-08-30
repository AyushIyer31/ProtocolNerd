#!/usr/bin/env python3
"""Write the static frontend config from deployment environment variables."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_JS = ROOT / "protocolnerd-website" / "env.js"
DEFAULT_API_URL = "http://localhost:8001"


def main() -> None:
    api_url = (
        os.getenv("PROTOCOLSNERD_API_URL")
        or os.getenv("PROTOCOLSNERD_BACKEND_URL")
        or os.getenv("BACKEND_URL")
        or os.getenv("API_URL")
        or DEFAULT_API_URL
    ).strip().rstrip("/")

    contents = f"""(function () {{
    const configuredApiUrl = {json.dumps(api_url)};

    window.env = {{
        FRONTEND_FLOW: {{
            SITE_NAME: "ProtocolsNerd",
            SITE_LOGO: "assets/protocols_nerd_default_logo.png",
            SITE_ICON: "CN",
            SITE_TAGLINE: "Local AI document analysis with agentic and prompt-based execution modes.",
            DISCLAIMER: "This tool performs local document analysis using Ollama. No data leaves your machine. Results should be reviewed by a qualified professional.",
            QUESTION_PLACEHOLDER: "Example: Check whether this interconnection agreement complies with the uploaded regulations.",
            STYLES: {{
                BACKGROUND_COLOR: "#EFF8FF",
                FONT_FAMILY: "'Roboto', sans-serif",
                SUBMIT_BUTTON_BG: "#007bff"
            }},
            API_URL: configuredApiUrl
        }}
    }};
}})();
"""

    ENV_JS.write_text(contents, encoding="utf-8")
    print(f"Wrote {ENV_JS.relative_to(ROOT)} with API_URL={api_url}")


if __name__ == "__main__":
    main()
