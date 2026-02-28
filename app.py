from __future__ import annotations

from data_analysis.runtime_app import *  # noqa: F403
from data_analysis.app_factory import create_app, run_from_env

app = create_app()

if __name__ == "__main__":
    run_from_env(app)
