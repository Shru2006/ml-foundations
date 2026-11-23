# ML Foundations

**The building blocks I practised before the lending AI series: data cleaning, regression and classification, one notebook per technique.**

> **Earlier learning work (2023).** These notebooks were written while following an online machine learning course, and several follow its templates closely. For my current, end-to-end work, see the [lending AI series](https://github.com/Shru2006).

`Python` · `pandas` · `scikit-learn` · `matplotlib` · `Jupyter`

This repo merges three earlier repos (`Basics`, `Regression`, `Classification`), which are now archived. Everything uses small public teaching datasets. Nothing here is financial advice.

---

## What's inside

### `basics/`: getting data ready

| Notebook | Data | What it covers |
|---|---|---|
| [01 Loading and cleaning data](basics/01_loading_and_cleaning_data.ipynb) | US crude oil production by state (monthly) | Loading, inspecting, fixing types and gaps |
| [02 Exploring and visualising relationships](basics/02_exploring_and_visualising_relationships.ipynb) | Processed crude oil data | Plots and relationships between variables |
| [03 Missing data and encoding](basics/03_missing_data_and_encoding.ipynb) | 10-row customer table | Imputing missing values, one-hot and label encoding |

### `regression/`: predicting a number

| Notebook | Data | Technique |
|---|---|---|
| [01 Simple linear regression](regression/01_simple_linear_regression.ipynb) | Salary vs years of experience | One feature, straight line |
| [02 Multiple linear regression](regression/02_multiple_linear_regression.ipynb) | 50 start-ups' spend and profit | Several features, encoded categories |
| [03 Polynomial regression](regression/03_polynomial_regression.ipynb) | Salary by position level | Curved fit vs straight line |
| [04 Support vector regression](regression/04_support_vector_regression.ipynb) | Salary by position level | Feature scaling + SVR |
| [05 Decision tree regression](regression/05_decision_tree_regression.ipynb) | Salary by position level | Tree-based, step-shaped fit |
| [06 Random forest regression](regression/06_random_forest_regression.ipynb) | Salary by position level | Ensemble of trees |

### `classification/`: predicting a yes/no

All four use the same data (age and salary → bought a product or not), so the models can be compared like for like.

| Notebook | Technique |
|---|---|
| [01 Logistic regression](classification/01_logistic_regression.ipynb) | Linear decision boundary |
| [02 K-nearest neighbours](classification/02_k_nearest_neighbours.ipynb) | Distance-based |
| [03 Support vector machine](classification/03_support_vector_machine.ipynb) | Maximum-margin boundary |
| [04 Decision tree classification](classification/04_decision_tree_classification.ipynb) | Rule-based splits |

## Fixes made when these were reorganised (2026)

**Bugs.** Three notebooks ended in errors or quietly gave wrong results:

| Notebook | Fix |
|---|---|
| Polynomial regression | The evaluation cell used `y_test` and `y_pred`, which this notebook never creates (the 10-row dataset isn't split). It now reports in-sample R² for both models: 0.67 for the straight line, 0.92 for the polynomial. |
| K-nearest neighbours | The test set was rescaled with its own statistics (`fit_transform`), leaking test information. It now uses the scaler fitted on the training data, and a typo in the test-set plot is fixed. Test accuracy: 93%. |
| Support vector machine | The train/test split unpacked its outputs in the wrong order, so the model trained on the wrong arrays. Fixed, along with the same scaling leak, and the two empty plot sections are completed. Test accuracy: 90%. |

**Compatibility.** Four small changes let the notebooks run on current pandas, NumPy and scikit-learn. None of them changes the method:
- `np.arange(min(x), max(x))` became `np.arange(x.min(), x.max())` in four regression notebooks.
- In the support vector regression notebook, predictions are reshaped to 2-D before `inverse_transform`.
- In the crude-oil notebook, the yearly `groupby(...).sum()` skips the date column (`numeric_only=True`).

**Verified:** 11 of the 13 notebooks were re-run end to end on current library versions with no errors. Logistic regression and decision tree classification had no errors, and keep their original outputs: their fine-grained plotting grids take a very long time to compute.

A duplicate support vector machine notebook, a copy of the course template, was dropped. The originals remain in the archived repos' history.

## Run it yourself

```bash
git clone https://github.com/Shru2006/ml-foundations.git
cd ml-foundations
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open a notebook from inside its folder (for example `regression/`). Each one reads its data from that folder's `data/`.

## Structure

```
├── basics/           3 notebooks + data/
├── regression/       6 notebooks + data/
├── classification/   4 notebooks + data/
└── requirements.txt
```

---

**Author:** Shruti Prabhu · [LinkedIn](https://www.linkedin.com/in/shruti-prabhu-32b77477) · [GitHub](https://github.com/Shru2006)
