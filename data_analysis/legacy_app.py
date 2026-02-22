"""Backward-compatibility shim.

Runtime now lives in ``data_analysis.runtime_app``. This module remains only to
avoid breaking older direct imports of ``data_analysis.legacy_app``.
"""

from data_analysis.runtime_app import *  # noqa: F401,F403

