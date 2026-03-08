# MSIS 522 HW1 - Data Science Workflow

This repository now includes:

- `MSIS_522_HW1_Sraddha.ipynb` - main notebook analysis
- `train_and_export.py` - trains all required models and exports saved artifacts
- `streamlit_app.py` - required 4-tab Streamlit app with interactive prediction
- `requirements.txt` - Python dependencies
- `artifacts/` - generated models, plots, metrics (created after training)

## Requirement check (from assignment PDF)

### Covered in notebook
- Part 1: Dataset intro, target distribution, multiple visualizations, correlation heatmap
- Part 2: Logistic baseline, Decision Tree CV, Random Forest CV, LightGBM CV, Neural Network, model comparison
- Part 3: SHAP summary/bar/waterfall + interpretation

### Was missing and now added
- Part 4 Streamlit deployment implementation (`streamlit_app.py`)
- Pre-trained model export pipeline (`train_and_export.py`)
- Reproducibility files (`requirements.txt`, this `README.md`)

## Run steps

1. Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

2. Make sure `covid.csv` is present in this folder.

3. Train models and export artifacts:
```bash
python3 train_and_export.py --data-path covid.csv
```

4. Launch Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Expected artifact outputs

After step 3, `artifacts/` will contain:
- `models/` - saved models (`.joblib` + `keras_mlp.keras`)
- `plots/` - descriptive, ROC, SHAP, and comparison plots
- `meta/` - metrics tables, best hyperparameters, summary JSON, feature schema

## Deploy

Push the repo to GitHub, then deploy `streamlit_app.py` on Streamlit Community Cloud.

### Deployed App (Public URL)

https://msis522hw1-sraddha.streamlit.app/
