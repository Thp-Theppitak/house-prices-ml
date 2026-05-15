# House Prices ML 🏠

Machine Learning pipeline for predicting house prices using the Ames Housing dataset.

## Results
| Model | Test RMSE |
|-------|-----------|
| Random Forest (tuned) | $29,508 |
| Linear Regression | $38,001 |

## What I did
- EDA on 1,460 houses with 81 features
- Handled missing values (19 columns)
- Selected 14 best features using correlation analysis
- Tuned hyperparameters with GridSearchCV (900 combinations, 10-fold CV)

## Tech Stack
Python · Scikit-learn · Pandas · Matplotlib

## Key Findings
- `OverallQual` correlates most with price (0.79)
- GridSearchCV improved RMSE from $32,338 → $29,508
