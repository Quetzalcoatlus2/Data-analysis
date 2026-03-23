from __future__ import annotations

from flask import render_template, request


def _research_context(filename: str) -> dict[str, str]:
    display_name = (request.args.get("display") or "").strip() or filename
    selected_col = (request.args.get("column") or request.args.get("selected_col") or "").strip()
    return {
        "filename": filename,
        "display_name": display_name,
        "selected_col": selected_col,
    }


def handle_research_hub(filename: str):
    return render_template("research_hub.html", **_research_context(filename))


def handle_research_forecast_lab(filename: str):
    return render_template("research_forecast.html", **_research_context(filename))


def handle_research_anomaly_lab(filename: str):
    return render_template("research_anomaly.html", **_research_context(filename))


def handle_research_quality_lab(filename: str):
    return render_template("research_quality.html", **_research_context(filename))


def handle_research_change_point_lab(filename: str):
    return render_template("research_change_point.html", **_research_context(filename))


def handle_research_conformal_lab(filename: str):
    return render_template("research_conformal.html", **_research_context(filename))


def handle_research_shap_lab(filename: str):
    return render_template("research_shap.html", **_research_context(filename))


def handle_research_multivariate_lab(filename: str):
    return render_template("research_multivariate.html", **_research_context(filename))


__all__ = [
    "handle_research_hub",
    "handle_research_forecast_lab",
    "handle_research_anomaly_lab",
    "handle_research_quality_lab",
    "handle_research_change_point_lab",
    "handle_research_conformal_lab",
    "handle_research_shap_lab",
    "handle_research_multivariate_lab",
]
