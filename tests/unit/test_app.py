
import sys
import os

# Add project root to sys.path to ensure app can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app import allowed_file, app

def test_allowed_file():
    """Test the allowed_file function."""
    assert allowed_file("data.csv") is True
    assert allowed_file("data.txt") is True
    assert allowed_file("data.xlsx") is True
    assert allowed_file("data.json") is True
    assert allowed_file("image.png") is False
    assert allowed_file("script.py") is False
    assert allowed_file("data") is False

def test_app_config():
    """Test basic app configuration."""
    assert app.config['UPLOAD_FOLDER'] == 'datasets'
    assert app.config['SECRET_KEY'] is not None

def test_index_route():
    """Test that the index route returns 200."""
    with app.test_client() as client:
        response = client.get('/')
        assert response.status_code == 200
        # Check for expected content
        assert b"Upload Dataset" in response.data or b"Data Analysis" in response.data
