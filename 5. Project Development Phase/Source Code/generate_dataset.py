"""
generate_dataset.py
--------------------
Generates a realistic, synthetic Human Development Index (HDI) dataset
(HDI.csv) for use in this project.

NOTE:
This dataset is SYNTHETICALLY GENERATED for educational / academic project
purposes. It is statistically realistic (feature ranges and relationships
mirror real-world UNDP HDI patterns) but the country-level numbers are not
official UNDP figures. For a real submission, you can replace HDI.csv with
the official dataset from https://hdr.undp.org/data-center while keeping
the same column names, and the rest of the pipeline (notebook, training
script, Flask app) will continue to work without any changes.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

COUNTRIES = [
    "Norway", "Switzerland", "Ireland", "Germany", "Australia", "Iceland",
    "Sweden", "Netherlands", "Denmark", "Finland", "Belgium", "New Zealand",
    "Canada", "Liechtenstein", "United Kingdom", "Japan", "South Korea",
    "United States", "Israel", "Luxembourg", "France", "Austria", "Spain",
    "Italy", "Slovenia", "Czech Republic", "Portugal", "Greece", "Poland",
    "Estonia", "Chile", "Argentina", "Croatia", "Hungary", "Malaysia",
    "Russia", "Turkey", "Bulgaria", "Mexico", "Brazil", "Romania",
    "China", "Thailand", "Sri Lanka", "Colombia", "Peru", "Ukraine",
    "Vietnam", "Philippines", "Indonesia", "Egypt", "South Africa",
    "Morocco", "India", "Bangladesh", "Kenya", "Ghana", "Cambodia",
    "Nepal", "Myanmar", "Pakistan", "Zambia", "Nigeria", "Rwanda",
    "Tanzania", "Uganda", "Senegal", "Ethiopia", "Haiti", "Yemen",
    "Sudan", "Afghanistan", "Chad", "Niger", "Mali", "Burkina Faso",
    "Mozambique", "Sierra Leone", "Burundi", "South Sudan", "Guinea",
    "Central African Republic", "Malawi", "Congo (DRC)", "Madagascar",
    "Liberia", "Togo", "Benin", "Gambia", "Guinea-Bissau", "Comoros",
    "Angola", "Cameroon", "Zimbabwe", "Lesotho", "Papua New Guinea",
    "Bolivia", "Guatemala", "Honduras", "Nicaragua", "Paraguay",
    "Ecuador", "Venezuela", "Jamaica", "Dominican Republic", "Panama",
    "Costa Rica", "Uruguay", "Cuba", "Jordan", "Lebanon", "Tunisia",
    "Algeria", "Iraq", "Iran", "Saudi Arabia", "United Arab Emirates",
    "Qatar", "Kuwait", "Bahrain", "Oman", "Kazakhstan", "Uzbekistan",
    "Georgia", "Armenia", "Azerbaijan", "Mongolia", "North Macedonia",
    "Serbia", "Bosnia and Herzegovina", "Albania", "Moldova", "Belarus",
    "Lithuania", "Latvia", "Slovakia", "Cyprus", "Malta", "Singapore",
    "Hong Kong", "Brunei", "Fiji", "Samoa", "Tonga", "Botswana",
    "Namibia", "Gabon", "Eswatini", "Djibouti", "Mauritania",
]

n = len(COUNTRIES)


def make_feature(base_hdi, spread, low, high):
    """Generate a feature correlated with a hidden HDI 'true score'."""
    val = base_hdi * (high - low) + low
    val += np.random.normal(0, spread, size=n)
    return np.clip(val, low, high)


# Hidden "true" development score per country (0-1), used to correlate features
true_score = np.random.beta(2.2, 2.0, size=n)
true_score = np.sort(true_score)[::-1]           # sort descending
np.random.shuffle(true_score[10:])                # keep top slightly ordered, shuffle rest a bit
# re-sort a softened version so top countries generally land near top
true_score = 0.6 * np.sort(true_score)[::-1] + 0.4 * np.random.permutation(true_score)

life_expectancy = make_feature(true_score, 3.0, 50, 85)
mean_years_schooling = make_feature(true_score, 1.2, 1.5, 13.5)
expected_years_schooling = make_feature(true_score, 1.5, 4.0, 20.0)

# GNI per capita is highly skewed -> generate on log scale then exponentiate
log_gni = make_feature(true_score, 0.6, np.log(500), np.log(120000))
gni_per_capita = np.round(np.exp(log_gni), 2)

life_expectancy = np.round(life_expectancy, 1)
mean_years_schooling = np.round(mean_years_schooling, 1)
expected_years_schooling = np.round(expected_years_schooling, 1)

# Compute a composite index (loosely mirrors UNDP HDI formula, normalized 0-1)
le_index = (life_expectancy - 20) / (85 - 20)
ms_index = mean_years_schooling / 15
es_index = expected_years_schooling / 18
edu_index = (ms_index + es_index) / 2
gni_index = (np.log(gni_per_capita) - np.log(100)) / (np.log(150000) - np.log(100))

hdi_score = (le_index * edu_index * gni_index) ** (1 / 3)
hdi_score = np.clip(hdi_score, 0, 1)


def categorize(score):
    if score >= 0.800:
        return "Very High"
    elif score >= 0.700:
        return "High"
    elif score >= 0.550:
        return "Medium"
    else:
        return "Low"


hdi_category = [categorize(s) for s in hdi_score]

df = pd.DataFrame({
    "Country": COUNTRIES,
    "Life Expectancy": life_expectancy,
    "Mean Years of Schooling": mean_years_schooling,
    "Expected Years of Schooling": expected_years_schooling,
    "GNI Per Capita": gni_per_capita,
    "HDI Score": np.round(hdi_score, 3),
    "HDI Category": hdi_category,
})

# Introduce a few missing values and duplicate rows to make the EDA/cleaning
# sections of the notebook meaningful (mirrors real messy data).
rng = np.random.default_rng(7)
missing_idx = rng.choice(df.index, size=6, replace=False)
missing_cols = rng.choice(
    ["Life Expectancy", "Mean Years of Schooling", "Expected Years of Schooling", "GNI Per Capita"],
    size=6,
)
for idx, col in zip(missing_idx, missing_cols):
    df.loc[idx, col] = np.nan

duplicate_rows = df.sample(3, random_state=1)
df = pd.concat([df, duplicate_rows], ignore_index=True)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("HDI.csv", index=False)
print(f"HDI.csv generated with {len(df)} rows.")
print(df["HDI Category"].value_counts())
