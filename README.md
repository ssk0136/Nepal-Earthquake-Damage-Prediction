# Nepal Earthquake Building Damage Prediction

Machine learning mini-project that predicts building damage levels using a **Random Forest Classifier** and building records from the 2015 Nepal earthquake.

## Objective

Predict one of three damage levels from a building's location and structural characteristics:

- **1:** Low damage
- **2:** Moderate damage
- **3:** Severe damage (near-complete destruction)

This predicts recorded damage classes; it does **not** predict when earthquakes will occur.

## Dataset

Source: [DrivenData — Richter's Predictor: Modeling Earthquake Damage](https://www.drivendata.org/competitions/57/nepal-earthquake/)

Download these files from the competition's **Data** page (sign-in/join may be required):

- `train_values.csv` — 260,601 building records with 38 input features and a `building_id` column.
- `train_labels.csv` — corresponding `building_id` and `damage_grade` (1, 2, or 3).

The files are joined using `building_id`. They are not included in this repository; download them from the source above.

## Implementation

The project uses **Python in Google Colab** and the following libraries:

- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- joblib

Workflow:

1. Load and merge the two CSV files.
2. Separate input features and the damage label; exclude `building_id` from inputs.
3. Convert categorical inputs with one-hot encoding (`pd.get_dummies`).
4. Make a stratified **80% training / 20% test** split (`random_state=42`).
5. Train a `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`.
6. Evaluate on held-out test data with accuracy, micro F1-score, a classification report and a confusion matrix.
7. Predict the damage class and estimated class probabilities for a selected test building.
8. Create plots for damage distribution, confusion matrix and feature importance; save the trained model.

## How to run

1. Open [Google Colab](https://colab.research.google.com/).
2. Choose **File → Upload notebook** and select `Nepal_Earthquake_Damage_Prediction_Clean.ipynb` from this repository.
3. Download `train_values.csv` and `train_labels.csv` from DrivenData to your computer.
4. In Colab, choose **Runtime → Run all** (or run cells from top to bottom).
5. When the file upload prompt appears, select both CSV files.
6. Wait for training to finish and inspect the printed results, sample prediction and three plots.

**Note:** Colab files are temporary. Upload the CSVs again if the runtime resets. Training time depends on the runtime. The notebook produces `nepal_earthquake_model.pkl`, `damage_distribution.png`, `confusion_matrix.png`, and `feature_importance.png` in the current session; download them from Colab's Files panel if needed.

## Results

The original Random Forest run on our **52,121-row held-out test split** produced:

| Metric | Result |
| --- | ---: |
| Accuracy | 71.60% |
| Micro F1-score | approximately 0.716 |
| Macro F1-score | 0.66 |
| Weighted F1-score | 0.71 |

The model did best on moderate damage (Grade 2), while severely damaged buildings were sometimes predicted as moderately damaged. Exact results should be checked by rerunning the notebook; they can vary with software versions.

The original reference paper reports a **0.7150 micro F1-score** for Random Forest, evaluated separately. This is not a direct benchmark comparison because the evaluation splits may differ.

## Limitations

- Damage grades are learned from historical Nepal earthquake survey data; predictions may not generalize to another earthquake or location.
- Grade 1 and Grade 3 are harder for the model to recognize than Grade 2.
- Feature importance reflects the model's predictive use of features, not causal proof.
- Model class probabilities are estimates, not calibrated certainty.

## Reference

Yitian Liang (2021), *Application of Machine Learning Methods to Predict the Level of Buildings Damage in the Nepal Earthquake*, Stanford CS229 Final Project Report.

Dataset and competition: [Richter's Predictor — DrivenData](https://www.drivendata.org/competitions/57/nepal-earthquake/).
