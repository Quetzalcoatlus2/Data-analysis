from data_analysis.ai import engine as ai_engine


def test_normalize_model_aliases_prefers_stable_flash_lite_over_preview() -> None:
    candidates = ai_engine._normalize_model_aliases("gemini-3.1-flash-lite-preview")

    assert "gemini-3.1-flash-lite" in candidates
    assert "gemini-3.1-flash-lite-preview" in candidates
    assert candidates.index("gemini-3.1-flash-lite") < candidates.index(
        "gemini-3.1-flash-lite-preview"
    )


def test_normalize_model_aliases_omits_preview_fallbacks() -> None:
    candidates = ai_engine._normalize_model_aliases("gemini-2.5-flash-lite")

    assert "gemini-3.1-flash-lite" in candidates
    assert "gemini-3.1-flash-lite-preview" not in candidates
    assert "models/gemini-3.1-flash-lite-preview" not in candidates
