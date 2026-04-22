from flask import Blueprint

from data_analysis.reports.pdf_report import handle_download_full_report_pdf
from data_analysis.routes.api import (
    handle_api_ai_summary,
    handle_api_interactive_data,
    handle_api_labs_data,
    handle_api_labs_meta,
    handle_full_history_json,
    handle_health,
)
from data_analysis.routes.downloads import (
    handle_download_ai_summary_html,
    handle_download_cleaned_csv,
    handle_download_full_report_html,
    handle_download_static_plots_zip,
)

api_bp = Blueprint('api', __name__)

@api_bp.route('/health', methods=['GET'])
def health():
    return handle_health()

@api_bp.route('/api/ai-summary/<filename>', methods=['GET'])
def api_ai_summary(filename):
    return handle_api_ai_summary(filename)

@api_bp.route('/api/interactive/<filename>', methods=['GET'])
def api_interactive_data(filename):
    return handle_api_interactive_data(filename)


@api_bp.route('/api/labs/<filename>/meta', methods=['GET'])
def api_labs_meta(filename):
    return handle_api_labs_meta(filename)


@api_bp.route('/api/labs/<filename>/<lab_key>', methods=['GET'])
def api_labs_data(filename, lab_key):
    return handle_api_labs_data(filename, lab_key)

@api_bp.route('/download/<filename>/cleaned.csv', methods=['GET'])
def download_cleaned_csv(filename):
    return handle_download_cleaned_csv(filename)

@api_bp.route('/download/<filename>/ai_summary.html', methods=['GET'])
def download_ai_summary_html(filename):
    return handle_download_ai_summary_html(filename)

@api_bp.route('/download/<filename>/static_plots.zip', methods=['GET'])
def download_static_plots_zip(filename):
    return handle_download_static_plots_zip(filename)

@api_bp.route('/download/<filename>/report.pdf', methods=['GET'])
def download_full_report_pdf(filename):
    return handle_download_full_report_pdf(filename)

@api_bp.route('/download/<filename>/report.html', methods=['GET'])
def download_full_report_html(filename):
    return handle_download_full_report_html(filename)

@api_bp.route('/full_history_json', methods=['GET'])
def full_history_json():
    return handle_full_history_json()
