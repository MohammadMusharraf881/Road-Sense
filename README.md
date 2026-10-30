# RoadSense — Road Collision Analytics & ML Dashboard

RoadSense is a straightforward Python data-science / machine-learning portfolio project. It uses pandas for data preparation, scikit-learn for a Random Forest classifier, Plotly for interactive visualizations, and Streamlit for the dashboard. It runs as a single Python application and needs no separate model service.

## Features

- Executive overview with KPIs, severity mix, monthly trend, speed-limit and hourly charts
- Interactive filters for severity, year, and area type
- Road, weather, surface, and weekday pattern exploration
- Geographic collision map using available coordinates
- Random Forest model evaluation with held-out accuracy, macro/weighted F1, confusion matrix, and feature importance
- Interactive scenario form showing model-estimated class probabilities
- Filterable data explorer with CSV download and missing-value inspection

## Requirements

- Python 3.10–3.12 recommended
- Internet connection for first-time package installation and map tiles

## Run on Windows

1. Extract the ZIP and open Command Prompt in the project folder (the folder containing `requirements.txt`).
2. Create and activate an environment:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```

3. Install packages:

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Start the dashboard:

```bat
streamlit run dashboard/app.py
```

5. Open the local address printed by Streamlit, normally `http://localhost:8501`.

Run tests with:

```bat
python -m pytest tests -v
```

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Open https://share.streamlit.io/ and create a new app.
3. Select your repository and branch.
4. Set the main file path to `dashboard/app.py`.
5. Deploy. Streamlit installs packages from `requirements.txt`.

Keep `data/collision-2025.csv` in the repository. The app reads this source file and trains a modest reproducible model on a stratified sample the first time the ML page is opened. No saved model artifacts are required.

## Dataset and interpretation

The supplied source file is `data/collision-2025.csv`. Collision severity labels are mapped as 1 = Fatal, 2 = Serious, and 3 = Slight. Other source fields such as road type and weather are numeric source codes; consult the dataset's official codebook for exact descriptions.

The model is an educational analysis of historical records. Its scenario estimate is not a real-time crash prediction and must not be used for driving, emergency response, insurance, or enforcement decisions. Correlations in historical data do not establish causation.
