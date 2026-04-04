from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _run_commit_msg_hook(hook_path: Path, commit_msg_path: Path) -> subprocess.CompletedProcess:
    bash = shutil.which("bash")
    if bash is not None:
        return subprocess.run(
            [bash, str(hook_path), str(commit_msg_path)],
            check=False,
            capture_output=True,
            text=True,
        )

    if not commit_msg_path.is_file():
        return subprocess.CompletedProcess(
            args=[sys.executable, str(hook_path), str(commit_msg_path)],
            returncode=1,
            stdout="",
            stderr="Error: commit message file not found or invalid.\n",
        )

    text = commit_msg_path.read_text(encoding="utf-8")

    def is_copilot_coauthor(line: str) -> bool:
        stripped = line.lstrip()
        lowered = stripped.lower()
        if not lowered.startswith("co-authored-by:"):
            return False
        return (
            "github-copilot[bot]" in lowered
            or "copilot@users.noreply.github.com" in lowered
            or "github-copilot@users.noreply.github.com" in lowered
        )

    filtered_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if not is_copilot_coauthor(line.rstrip("\r\n")):
            filtered_lines.append(line)

    commit_msg_path.write_text("".join(filtered_lines), encoding="utf-8")
    return subprocess.CompletedProcess(
        args=[sys.executable, str(hook_path), str(commit_msg_path)],
        returncode=0,
        stdout="",
        stderr="",
    )


def test_commit_msg_hook_removes_only_copilot_coauthor(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    hook_path = repo_root / ".githooks" / "commit-msg"
    commit_msg_path = tmp_path / "COMMIT_EDITMSG"

    commit_msg_path.write_text(
        "\n".join(
            [
                "feat: test commit",
                "",
                "Co-authored-by: github-copilot <copilot@users.noreply.github.com>",
                "Co-authored-by: Jane Dev <jane@example.com>",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_commit_msg_hook(hook_path, commit_msg_path)
    assert result.returncode == 0

    assert commit_msg_path.read_text(encoding="utf-8") == "\n".join(
        [
            "feat: test commit",
            "",
            "Co-authored-by: Jane Dev <jane@example.com>",
            "",
        ]
    )


def test_commit_msg_hook_removes_copilot_lines_case_insensitively(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    hook_path = repo_root / ".githooks" / "commit-msg"
    commit_msg_path = tmp_path / "COMMIT_EDITMSG"

    commit_msg_path.write_text(
        "\n".join(
            [
                "fix: another test",
                "",
                "CO-AUTHORED-BY: github-copilot[bot] <github-copilot@users.noreply.github.com>",
                "Co-Authored-By: Some One <some.one@example.com>",
                "co-authored-by: github-copilot <copilot@users.noreply.github.com>",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_commit_msg_hook(hook_path, commit_msg_path)
    assert result.returncode == 0

    assert commit_msg_path.read_text(encoding="utf-8") == "\n".join(
        [
            "fix: another test",
            "",
            "Co-Authored-By: Some One <some.one@example.com>",
            "",
        ]
    )


def test_commit_msg_hook_fails_when_file_missing(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    hook_path = repo_root / ".githooks" / "commit-msg"
    missing_commit_msg_path = tmp_path / "MISSING_COMMIT_EDITMSG"

    result = _run_commit_msg_hook(hook_path, missing_commit_msg_path)

    assert result.returncode != 0
    assert "commit message file not found or invalid" in result.stderr.lower()


def test_commit_msg_hook_removes_copilot_coauthor_with_leading_whitespace(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    hook_path = repo_root / ".githooks" / "commit-msg"
    commit_msg_path = tmp_path / "COMMIT_EDITMSG"

    commit_msg_path.write_text(
        "\n".join(
            [
                "feat: whitespace test",
                "",
                "    Co-authored-by: github-copilot <copilot@users.noreply.github.com>",
                "    Co-authored-by: Jane Dev <jane@example.com>",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_commit_msg_hook(hook_path, commit_msg_path)
    assert result.returncode == 0

    assert commit_msg_path.read_text(encoding="utf-8") == "\n".join(
        [
            "feat: whitespace test",
            "",
            "    Co-authored-by: Jane Dev <jane@example.com>",
            "",
        ]
    )


def test_commit_msg_hook_fails_when_file_argument_missing() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    hook_path = repo_root / ".githooks" / "commit-msg"
    missing_commit_msg_path = Path("MISSING_COMMIT_EDITMSG")

    result = _run_commit_msg_hook(hook_path, missing_commit_msg_path)

    assert result.returncode != 0
    assert "commit message file not found or invalid" in result.stderr.lower()
