# Athlete Injury Prediction Model

A machine learning project that predicts injury risk for collegiate athletes based on training load, recovery, fatigue, and performance data. The full workflow — data exploration, preprocessing, model training, hyperparameter tuning, and a risk-level prediction function — is in [`Sports_Injury_Prediction_ML_Model.ipynb`](Sports_Injury_Prediction_ML_Model.ipynb).

## Dataset

`collegiate_athlete_injury_dataset.csv` — 200 athletes, 17 columns.

- **Target:** `Injury_Indicator` (0 = no injury, 1 = injury). The classes are heavily imbalanced: 186 non-injured vs 14 injured (7%).
- **Features:** Age, Height, Weight, Gender, Position, Training Intensity, Training Hours/Week, Recovery Days/Week, Match Count/Week, Rest Between Events, Fatigue Score, Performance Score, Team Contribution Score, Load Balance Score, ACL Risk Score.

## Approach

1. **EDA** — injury distribution, fatigue vs. injury box plots, correlation heatmap.
2. **Preprocessing** — dropped `Athlete_ID`, one-hot encoded `Gender` and `Position`, standardized numeric features with `StandardScaler`.
3. **Modeling** — Logistic Regression and Decision Tree, both with `class_weight='balanced'` to handle the class imbalance (80/20 stratified train/test split).
4. **Tuning** — `GridSearchCV` (10-fold CV) over the regularization strength `C`, optimizing for **recall** since missing an injured athlete is costlier than a false alarm. Best value: `C = 0.01`.
5. **Deployment helpers** — a `predict_injury_risk()` function that maps predicted probability to Low / Medium / High risk levels, and the final model saved as `injury_model.pkl` with joblib.

## Results

Evaluated on a held-out test set of 40 athletes (37 non-injured, 3 injured):

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression (baseline, C=0.001) | 87.5% | 0.33 | 0.67 | 0.44 |
| Decision Tree | 87.5% | 0.00 | 0.00 | 0.00 |
| **Tuned Logistic Regression (C=0.01)** | **92.5%** | **0.50** | **1.00** | **0.67** |

### Final model (tuned Logistic Regression)

- **Test Accuracy:** 92.5%
- **Recall:** 1.00 — catches **all** injured athletes in the test set (0 false negatives)
- **Precision:** 0.50 — half of the flagged athletes were actually injured
- **F1 Score:** 0.67
- **Training Accuracy:** 90.0% vs Test Accuracy 92.5% — no overfitting

Confusion matrix (test set):

|  | Predicted: No Injury | Predicted: Injury |
|---|---|---|
| **Actual: No Injury** | 34 | 3 |
| **Actual: Injury** | 0 | 3 |

The Decision Tree reached 100% training accuracy but failed to identify any injured athletes on the test set (a classic sign of overfitting on imbalanced data), which is why the tuned Logistic Regression was chosen as the final model.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
jupyter notebook Sports_Injury_Prediction_ML_Model.ipynb
```

Run the cells top to bottom. The final cells demonstrate predicting the risk level for a new athlete and exporting the trained model to `injury_model.pkl`.
