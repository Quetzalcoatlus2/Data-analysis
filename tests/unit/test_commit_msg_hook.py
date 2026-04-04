from __future__ import annotations

import subprocess
from pathlib import Path


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

    subprocess.run(["bash", str(hook_path), str(commit_msg_path)], check=True)

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

    subprocess.run(["bash", str(hook_path), str(commit_msg_path)], check=True)

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

    result = subprocess.run(
        ["bash", str(hook_path), str(missing_commit_msg_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "commit message file not found or invalid" in result.stderr.lower()


def test_commit_msg_hook_fails_when_file_argument_missing() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    hook_path = repo_root / ".githooks" / "commit-msg"

    result = subprocess.run(
        ["bash", str(hook_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "commit message file not found or invalid" in result.stderr.lower()
