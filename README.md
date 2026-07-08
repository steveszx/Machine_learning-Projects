# Credit Card Fraud Classifier

A binary classification project on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), predicting whether a transaction is fraudulent from anonymized transaction features.

## Dataset

- 284,807 transactions, 492 of which are fraudulent (~0.17% positive class — highly imbalanced).
- Features `V1`-`V28` are PCA components (already scaled/anonymized); `Time` and `Amount` are raw and are scaled separately.
- Not included in this repo due to size — download `creditcard.csv` from the Kaggle link above and place it in the project root.

## Approach

1. **EDA** — checked distribution and skew of `Amount`/`Time`, and correlation of all features against the `Class` target to rule out any variable worth dropping.
2. **Preprocessing** — 70/30 train/test split; `RobustScaler` fit on `Time`/`Amount` in the training set only, then reused (not refit) to transform the test set, to avoid leaking test-set distribution info.
3. **Class imbalance** — handled with `SMOTE` inside an `imblearn` pipeline, so oversampling happens only on training folds.
4. **Models** — a hyperparameter-tuned `LogisticRegression` (L1, tuned via grid search) and a `RandomForestClassifier`, each wrapped in a `SMOTE + model` pipeline.

## Results

| Model | Precision (fraud) | Recall (fraud) | F1 (fraud) |
|---|---|---|---|
| Logistic Regression | 0.18 | 0.88 | 0.29 |
| Random Forest | 0.27 | 0.90 | 0.41 |

Both models catch the large majority of fraud cases (high recall), but Random Forest does so with meaningfully fewer false alarms, giving it the better precision/F1 tradeoff on this dataset.

## Running it

```bash
pip install -r requirements.txt
python credit_vers2.py
```
