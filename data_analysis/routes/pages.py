from flask import Blueprint

from data_analysis.routes.upload import handle_upload_file
from data_analysis.routes.analyze import handle_analyze_file

pages_bp = Blueprint('pages', __name__)

@pages_bp.route('/', methods=['GET', 'POST'])
def upload_file():
    return handle_upload_file()

@pages_bp.route('/analyze/<filename>', methods=['GET', 'POST'])
def analyze_file(filename):
    return handle_analyze_file(filename)
