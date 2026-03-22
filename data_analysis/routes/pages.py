from flask import Blueprint

from data_analysis.routes.analyze import handle_analyze_file
from data_analysis.routes.research import (
    handle_research_anomaly_lab,
    handle_research_change_point_lab,
    handle_research_conformal_lab,
    handle_research_forecast_lab,
    handle_research_hub,
    handle_research_multivariate_lab,
    handle_research_quality_lab,
    handle_research_shap_lab,
)
from data_analysis.routes.upload import handle_upload_file

pages_bp = Blueprint('pages', __name__)

@pages_bp.route('/', methods=['GET', 'POST'])
def upload_file():
    return handle_upload_file()

@pages_bp.route('/analyze/<filename>', methods=['GET', 'POST'])
def analyze_file(filename):
    return handle_analyze_file(filename)


@pages_bp.route('/labs/<filename>', methods=['GET'])
def labs_hub(filename):
    return handle_research_hub(filename)


@pages_bp.route('/labs/<filename>/forecast', methods=['GET'])
def labs_forecast_lab(filename):
    return handle_research_forecast_lab(filename)


@pages_bp.route('/labs/<filename>/anomaly', methods=['GET'])
def labs_anomaly_lab(filename):
    return handle_research_anomaly_lab(filename)


@pages_bp.route('/labs/<filename>/quality', methods=['GET'])
def labs_quality_lab(filename):
    return handle_research_quality_lab(filename)


@pages_bp.route('/labs/<filename>/change-points', methods=['GET'])
def labs_change_point_lab(filename):
    return handle_research_change_point_lab(filename)


@pages_bp.route('/labs/<filename>/conformal', methods=['GET'])
def labs_conformal_lab(filename):
    return handle_research_conformal_lab(filename)


@pages_bp.route('/labs/<filename>/shap', methods=['GET'])
def labs_shap_lab(filename):
    return handle_research_shap_lab(filename)


@pages_bp.route('/labs/<filename>/multivariate', methods=['GET'])
def labs_multivariate_lab(filename):
    return handle_research_multivariate_lab(filename)
