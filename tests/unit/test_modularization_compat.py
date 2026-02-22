import app as app_module
from data_analysis.app_factory import create_app


def test_endpoint_names_remain_stable():
    flask_app = create_app()
    endpoints = set(flask_app.view_functions.keys())
    expected = {
        "pages.upload_file",
        "pages.analyze_file",
        "api.health",
        "api.api_ai_summary",
        "api.api_interactive_data",
        "api.download_cleaned_csv",
        "api.download_ai_summary_html",
        "api.download_static_plots_zip",
        "api.download_full_report_pdf",
        "api.download_full_report_html",
        "api.full_history_json",
    }
    assert expected.issubset(endpoints)


def test_app_module_reexports_key_symbols():
    expected_symbols = [
        "app",
        "TinyLRU",
        "DATAFRAME_CACHE",
        "INTERACTIVE_DATA_CACHE",
        "NUMERIC_DF_CACHE",
        "AI_DESCRIBE_CACHE",
        "_compute_forecast",
        "generate_plot",
        "generate_forecast_plot",
        "detect_anomalies",
        "describe_for_ai",
        "get_cached_anomalies",
        "get_cached_numeric_df",
    ]
    for symbol in expected_symbols:
        assert hasattr(app_module, symbol), f"Missing compatibility export: {symbol}"
