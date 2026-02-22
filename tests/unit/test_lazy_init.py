import subprocess
import sys
from pathlib import Path


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_app_does_not_eager_load_legacy_runtime_or_genai():
    code = """
import sys
import app
assert 'data_analysis.legacy_app' not in sys.modules
assert 'google.generativeai' not in sys.modules
print('ok')
"""
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_loading_flask_app_still_defers_genai_and_ai_configure():
    code = """
import sys
from app import app
import data_analysis.legacy_app as legacy
assert 'data_analysis.legacy_app' in sys.modules
assert 'google.generativeai' not in sys.modules
assert legacy.AI_CONFIG_ATTEMPTED is False
print('ok')
"""
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
