# Student Learning Analytics System

A complete Python-based analytics solution for generating synthetic student learning data, performing data preprocessing, analyzing learning gaps, training performance models, and delivering personalized recommendations through a Streamlit dashboard.

## Project Structure

- `data/` - Generated CSV datasets and cleaned/enriched analytics files.
- `models/` - Persisted trained machine learning model artifacts.
- `notebooks/` - Exploratory data analysis notebooks.
- `src/` - Source modules for dataset generation, preprocessing, feature engineering, learning gap analysis, model training, and recommendation engine.
- `dashboard/` - Streamlit dashboard application.
- `requirements.txt` - Python package dependencies.
- `README.md` - Project overview and usage instructions.
- `architecture.png` - Architecture diagram placeholder.

## Setup

1. Create and activate a Python virtual environment.
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies.
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

### Phase 1: Generate synthetic datasets
```powershell
python src\generate_data.py
```

### Phase 2: Preprocess and clean data
```powershell
python src\preprocess.py
```

### Feature engineering and enrichment
```powershell
python src\feature_engineering.py
```

### Learning gap analysis
```powershell
python src\learning_gap.py
```

### Train machine learning models
```powershell
python src\train_model.py
```

### Create recommendations
```powershell
python src\recommendation.py
```

### Run Streamlit dashboard
```powershell
streamlit run dashboard\app.py
```

## Notes

- Generated datasets are stored in `data/`.
- Trained model artifacts are saved to `models/model.pkl`.
- The dashboard reads `student_analytics_enriched.csv` and `student_recommendations.csv`.
- Use the student search panel to inspect individual student profiles and recommendations.
