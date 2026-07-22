"""Launch Streamlit bound to Databricks Apps port."""

import os
import sys

from streamlit.web import cli as stcli


def main() -> None:
    port = os.environ.get("DATABRICKS_APP_PORT") or os.environ.get("PORT") or "8000"
    sys.argv = [
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(port),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
