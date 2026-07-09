
"""
test_predictions.py
--------------------
Sanity-check script: runs 10 representative example inputs through the
saved model pipeline and prints the predicted HDI Category next to the
expected category (based on realistic real-world reference points).

Run with:
    python test_predictions.py
"""

import joblib
import pandas as pd

FEATURE_COLUMNS = [
    "Life Expectancy",
    "Mean Years of Schooling",
    "Expected Years of Schooling",
    "GNI Per Capita",
]

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# (description, Life Expectancy, Mean Schooling, Expected Schooling, GNI, expected_category)
TEST_CASES = [
    ("Very high-income, high-longevity nation",       83.2, 12.9, 18.9, 68000, "Very High"),
    ("Advanced Western European economy",              81.5, 12.2, 17.5, 55000, "Very High"),
    ("Wealthy Gulf state",                              78.9, 10.3, 14.5, 62000, "High"),
    ("Upper-middle-income Latin American country",      75.0,  9.0, 14.8, 16000, "High"),
    ("Rapidly developing East Asian economy",           76.8,  8.4, 15.2, 20000, "High"),
    ("Emerging South-East Asian economy",               71.2,  7.5, 12.9,  8500, "Medium"),
    ("Lower-middle-income South Asian country",         69.5,  6.2, 11.5,  5200, "Medium"),
    ("Sub-Saharan African developing economy",           63.0,  5.0, 10.0,  3200, "Medium"),
    ("Low-income conflict-affected country",             56.5,  2.8,  6.5,  1100, "Low"),
    ("Least-developed country, limited infrastructure",  52.0,  2.1,  5.0,   750, "Low"),
]


def main():
    rows = []
    correct = 0
    for desc, le, ms, es, gni, expected in TEST_CASES:
        X = pd.DataFrame([[le, ms, es, gni]], columns=FEATURE_COLUMNS)
        X_scaled = scaler.transform(X)
        pred_encoded = model.predict(X_scaled)[0]
        predicted = label_encoder.inverse_transform([pred_encoded])[0]
        match = "YES" if predicted == expected else "NO"
        correct += predicted == expected
        rows.append([desc, le, ms, es, gni, expected, predicted, match])

    df = pd.DataFrame(
        rows,
        columns=[
            "Scenario", "Life Exp.", "Mean School.", "Exp. School.",
            "GNI", "Expected", "Predicted", "Match",
        ],
    )
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 140)
    print(df.to_string(index=False))
    print(f"\n{correct}/{len(TEST_CASES)} test cases matched the expected category.")
    print(
        "\nNote: 'Expected' values are realistic reference points, not guaranteed exact "
        "matches, since this project uses a synthetically generated training dataset. "
        "Swap in the official UNDP HDI.csv for production-grade accuracy."
    )


if __name__ == "__main__":
    main()
