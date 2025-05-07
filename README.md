# Titanic Survival Prediction

This repository contains an end-to-end machine learning pipeline for the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic), aimed at predicting passenger survival using Gradient Boosting.

## Project Structure

* `train.csv` — training dataset
* `test.csv` — test dataset for submission
* `gender_submission.csv` — example submission format
* `submission.csv` — model predictions for submission
* `Titanic_Housing_Displacement_Risk_Analysis.R` — (other project script, ignore for this task)

## Dependencies

The pipeline is implemented in Python and leverages the following libraries:

* pandas
* numpy
* scikit-learn
* imbalanced-learn (SMOTE)

In Colab, install dependencies via:

```bash
!pip install -q pandas numpy scikit-learn imbalanced-learn
```

## Data Preprocessing

1. **Feature Engineering**

   * Created `FamilySize` and `IsAlone` from `SibSp` and `Parch`.
   * Log-transformed `Fare` into `LogFare`.
   * Extracted passenger `Title` from `Name` and mapped to numerical codes.
2. **Missing Values**

   * Filled `Age` with median.
   * Filled `Embarked` with mode.
3. **Encoding & Cleanup**

   * One-hot encoded `Sex` and `Embarked`.
   * Dropped unused columns: `PassengerId`, `Name`, `Ticket`, `Cabin`, `Fare`.

## Model Training

* Applied SMOTE to balance classes.
* Standardized numerical features with `StandardScaler`.
* Trained a `GradientBoostingClassifier` with 200 trees, learning rate 0.05, max depth 3.
* Evaluated with 5-fold stratified cross-validation (accuracy \~0.844).

## Usage

1. Upload `train.csv`, `test.csv`, `gender_submission.csv` in Colab.
2. Run the training pipeline cell-by-cell.
3. Review the printed CV accuracy.
4. Download `submission.csv` and submit on Kaggle.

## Next Steps

* Tune hyperparameters (e.g., via grid search).
* Explore additional features (e.g., passenger cabin deck, family name).
* Compare alternate models (Random Forest, XGBoost).

---

*Riza Saireke — May 2025*
