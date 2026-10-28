import pytest
from streamlit.testing.v1 import AppTest
from src.core import ROOT


@pytest.fixture
def app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not at.exception
    return at


@pytest.mark.parametrize("page", ["predict", "explore", "models", "energy", "guide"])
def test_page_renders(app, page):
    app.switch_page(f"app_pages/{page}.py").run()
    assert not app.exception


def test_manual_prediction(app):
    app.switch_page("app_pages/predict.py").run()
    app.button[0].click().run()
    assert not app.exception
    assert any(m.label == "Occupied model score" for m in app.metric)
    app.selectbox(key="preset").select("Recorded empty example").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("empty" in item.value.lower() for item in app.info)


def test_filter_no_dates(app):
    app.switch_page("app_pages/explore.py").run()
    app.multiselect(key="explore_days").set_value([]).run()
    assert not app.exception
    assert any("Select at least" in item.value for item in app.info)


def test_model_views(app):
    app.switch_page("app_pages/models.py").run()
    for value in ["Sensor experiment", "Methodology"]:
        app.segmented_control(key="model_view").set_value(value).run()
        assert not app.exception


def test_energy_zero_power_and_delay(app):
    app.switch_page("app_pages/energy.py").run()
    app.number_input(key="energy_watts").set_value(0.0).run()
    assert not app.exception
    assert any(m.label == "Potential energy" and m.value == "0.00 kWh" for m in app.metric)
    app.slider(key="energy_delay").set_value(30).run()
    assert not app.exception
