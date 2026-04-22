import pytest

from app import app


@pytest.mark.parametrize(
    ("path", "page_key"),
    [
        ("/labs/{filename}", "hub"),
        ("/labs/{filename}/forecast", "forecast"),
        ("/labs/{filename}/anomaly", "anomaly"),
        ("/labs/{filename}/quality", "quality"),
        ("/labs/{filename}/change-points", "change-points"),
        ("/labs/{filename}/conformal", "conformal"),
        ("/labs/{filename}/shap", "shap"),
        ("/labs/{filename}/multivariate", "multivariate"),
    ],
)
def test_research_pages_render_runtime_shell(path: str, page_key: str):
    filename = "e" * 40 + ".csv"
    url = path.format(filename=filename) + "?display=demo&column=target"

    with app.test_client() as client:
        response = client.get(url)

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "id=\"labs-root\"" in html
    assert f'data-active-lab="{page_key}"' in html
    assert f'data-lab-page="{page_key}"' in html
    assert "research_labs.js" in html
    assert "research_labs.css" in html


def test_research_nav_links_preserve_column_selection_query_param():
    filename = "f" * 40 + ".csv"
    with app.test_client() as client:
        response = client.get(f"/labs/{filename}/forecast?display=demo&column=sales")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "column=sales" in html
