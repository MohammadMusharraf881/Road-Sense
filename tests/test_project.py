from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_exists_and_has_expected_columns():
    path = ROOT / 'data' / 'collision-2025.csv'
    assert path.exists()
    df = pd.read_csv(path, nrows=10)
    assert {'collision_severity', 'date', 'time', 'number_of_vehicles', 'number_of_casualties'} <= set(df.columns)


def test_severity_values_are_supported():
    df = pd.read_csv(ROOT / 'data' / 'collision-2025.csv', usecols=['collision_severity'])
    assert set(df['collision_severity'].dropna().unique()) <= {1, 2, 3}


def test_dashboard_uses_local_ml_stack():
    app = (ROOT / 'dashboard' / 'app.py').read_text(encoding='utf-8').lower()
    requirements = (ROOT / 'requirements.txt').read_text(encoding='utf-8').lower()
    assert 'randomforestclassifier' in app
    assert 'scikit-learn' in requirements
    assert 'streamlit' in requirements
    assert 'plotly' in requirements
