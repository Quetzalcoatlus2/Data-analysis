import pytest

from data_analysis.analysis.controls import (
    DEFAULT_FORECAST_PCT,
    forecast_steps_for_history,
    parse_contamination,
    parse_forecast_pct,
    resolve_data_range_selection,
)


def test_parse_forecast_pct_clamps_and_defaults():
    assert parse_forecast_pct(None) == pytest.approx(DEFAULT_FORECAST_PCT)
    assert parse_forecast_pct("") == pytest.approx(DEFAULT_FORECAST_PCT)
    assert parse_forecast_pct("bad") == pytest.approx(DEFAULT_FORECAST_PCT)
    assert parse_forecast_pct("-0.2") == pytest.approx(0.0)
    assert parse_forecast_pct("0.9") == pytest.approx(0.5)
    assert parse_forecast_pct("0.2") == pytest.approx(0.2)


def test_parse_contamination_clamps_and_defaults():
    assert parse_contamination(None, default=0.03) == pytest.approx(0.03)
    assert parse_contamination("bad", default=0.03) == pytest.approx(0.03)
    assert parse_contamination("0") == pytest.approx(0.001)
    assert parse_contamination("0.5") == pytest.approx(0.2)
    assert parse_contamination("0.075") == pytest.approx(0.075)


def test_forecast_steps_for_history_keeps_requested_visual_share():
    assert forecast_steps_for_history(0, 0.2) == 0
    assert forecast_steps_for_history(60, 0) == 0
    assert forecast_steps_for_history(60, 0.2) == 15
    assert forecast_steps_for_history(100, "0.05") == 5
    assert forecast_steps_for_history(100, 0.9) == 100


def test_resolve_data_range_selection_matches_route_semantics():
    ratio = resolve_data_range_selection("0.25", 100)
    assert ratio.requested == pytest.approx(0.25)
    assert ratio.ratio == pytest.approx(0.25)
    assert ratio.rows == 25

    rows = resolve_data_range_selection("30", 100)
    assert rows.requested == pytest.approx(30.0)
    assert rows.ratio == pytest.approx(0.3)
    assert rows.rows == 30

    full = resolve_data_range_selection("1.0", 100)
    assert full.requested == pytest.approx(1.0)
    assert full.ratio == pytest.approx(1.0)
    assert full.rows == 0

    invalid = resolve_data_range_selection("bad", 100)
    assert invalid.requested == pytest.approx(1.0)
    assert invalid.ratio == pytest.approx(1.0)
    assert invalid.rows == 0
