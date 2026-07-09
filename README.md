# HDI Atlas — Human Development Index Category Predictor

**A Comprehensive Measure of Well-Being: Human Development Index (HDI) Prediction**

A complete, end-to-end machine learning project that predicts a country's
**HDI Category** (`Low`, `Medium`, `High`, `Very High`) from four core
development indicators, wrapped in a Flask web application.

---

## Project Description

The **Human Development Index (HDI)** is a UNDP metric summarizing a
country's average achievement in three dimensions: a long and healthy life,
knowledge, and a decent standard of living. This project builds a
classification model that predicts which HDI bracket a country falls into,
given:

- **Life Expectancy** (years)
- **Mean Years of Schooling** (years)
- **Expected Years of Schooling** (years)
- **GNI Per Capita** (USD, PPP)

The prediction is served through a responsive Flask web app ("HDI Atlas")
with a live gauge visualization, batch CSV prediction, prediction history,
and dark/light mode.

> **Dataset note:** `HDI.csv` included in this repo is **synthetically
> generated** (see `generate_dataset.py`) to be statistically realistic for
> demonstration and coursework purposes. For production use, replace it
> with the official dataset from the
> [UNDP Human Development Reports data center](https://hdr.undp.org/data-center)
> — keep the same column names and every other file (notebook, training
> script, Flask app) works unchanged.

---

## Objectives

- Build a clean, reproducible ML pipeline: data → EDA → preprocessing →
  model comparison → evaluation → deployment.
- Compare multiple classification algorithms and automatically select the
  best performer.
- Package the model behind a polished, accessible web interface.
- Follow professional software engineering practices (clean functions,
  input validation, error handling, documentation).

---

## Dataset

| Column | Description |
|---|---|
| `Life Expectancy` | Average life expectancy at birth (years) |
| `Mean Years of Schooling` | Average years of schooling for adults 25+ |
| `Expected Years of Schooling` | Expected years of schooling for a child entering school |
| `GNI Per Capita` | Gross National Income per capita (USD, PPP) |
| `HDI Category` | Target label: `Low`, `Medium`, `High`, `Very High` |

`train_model.py` and `notebook.ipynb` **auto-detect the target column**, so
you can drop in a different HDI dataset with a slightly different column
name (e.g. `Category`) and everything keeps working.

---

## Technologies Used

- **Python 3.10+**
- **Flask** — web application framework
- **scikit-learn** — model training & evaluation
- **XGBoost** — gradient boosting (optional, auto-skipped if not installed)
- **pandas / numpy** — data handling
- **matplotlib / seaborn** — visualization
- **joblib** — model persistence
- **HTML5 / CSS3 / vanilla JavaScript** — frontend
- **Jupyter Notebook** — exploratory data analysis



## Installation

### 1. Clone / download the project

```bash
git clone https://github.com/vasanthikondreddy/HDI_prediction-.git
cd hdi_project
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Notebook

The dataset and notebook are already generated in this repo, but you can
rebuild them from scratch:

```bash
# 1. Regenerate the synthetic dataset (optional — HDI.csv is already included)
python generate_dataset.py

# 2. Rebuild notebook.ipynb from build_notebook.py
python build_notebook.py

# 3. Open it in Jupyter
jupyter notebook notebook.ipynb
```

The notebook contains dataset loading, cleaning, statistical summaries,
correlation analysis, distribution/box/pair plots, a heatmap, outlier
detection, preprocessing, model comparison, confusion matrix, ROC curve,
learning curve, cross-validation, and feature importance — each with
explanatory markdown.

---

## Training the Model

```bash
python train_model.py
```

This will:

1. Load and auto-clean `HDI.csv`.
2. Encode the target and scale the features.
3. Train **Logistic Regression, Decision Tree, Random Forest, Gradient
   Boosting, KNN, Naive Bayes, SVM**, and **XGBoost** (if installed).
4. Print Accuracy / Precision / Recall / F1 for every model.
5. Automatically select the best model by weighted F1 Score.
6. Save `model.pkl`, `scaler.pkl`, `label_encoder.pkl`, and
   `model_comparison.csv`.

---

## Running the Flask App

Make sure `model.pkl`, `scaler.pkl` and `label_encoder.pkl` exist (run
`train_model.py` first if not), then:

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**.

### Features available in the app

- **Predict** — enter the four indicators and get the predicted category,
  a confidence score, a plain-language meaning, a visual gauge, and
  improvement suggestions (for Medium/Low predictions).
- **Batch** — upload a CSV of many countries and get predictions for every
  row, with a downloadable results CSV.
- **History** — a running table + live bar chart of everything you've
  predicted this session.
- **Dark / Light mode** toggle.

---

## Testing

```bash
python test_predictions.py
```

Runs 10 representative example inputs (from very-high-income to
least-developed profiles) through the saved model and compares the
prediction against a realistic expected category. Example output:

```
                                       Scenario  Life Exp.  Mean School.  Exp. School.   GNI  Expected Predicted Match
        Very high-income, high-longevity nation       83.2          12.9          18.9 68000 Very High Very High   YES
              Advanced Western European economy       81.5          12.2          17.5 55000 Very High Very High   YES
     Upper-middle-income Latin American country       75.0           9.0          14.8 16000      High      High   YES
              ...
```



---

## Deployment

## Git Commands

```bash
git init                                
git add .                               
git commit -m "Initial commit"          
git branch -M main                      
git remote add origin https://github.com/vasanthikondreddy/HDI_prediction-.git      
git push -u origin main                 
```
**Render**
**Live link**:https://hdi-prediction-zrrq.onrender.com
**Live Demo link**:https://drive.google.com/file/d/1ZztP_s05NAoIX3CGT1ZKAIC9QVYDepT3/view?usp=sharing
## Contributors

### Vasanthi Kondreddy (Team Lead)

- Machine Learning Model Development
- Flask Backend Development
- Frontend Integration
- Model Training & Evaluation
- Documentation
- Deployment on Render
