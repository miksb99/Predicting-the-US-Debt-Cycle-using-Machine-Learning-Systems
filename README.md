# Machine Learning on the U.S. Debt Cycle

This repository contains code and data for the paper **“Machine Learning on the U.S. Debt Cycle” (2025) by Mikkel Shiffman**.  
The analysis combines Ray Dalio’s *Principles for Navigating Big Debt Crises*, Karsten Müller’s **Global Macro Database**, and current macro data (e.g., Trading Economics) to:

- Estimate where the U.S. currently sits on Ray Dalio’s Bubble/Depression gauge  
- Model the probability that **next year’s real GDP is negative** (a “depression” event)

If you have any questions, please reach out to **Mikkel.shiffmanborggren@yale.edu**.

---

## Contents

The repository is organized into two main components:

1. **DebtCycleGauge/** – Models for estimating the U.S. position on Ray Dalio’s Bubble/Depression gauge  
2. **DepressionGauge/** – Machine learning models for predicting the probability of next‑year depressions across countries

---

## 1. Approach & Key Findings

### 1.1 Ray Dalio’s Bubble/Depression Gauge

The goal is to map current macroeconomic conditions to Ray Dalio’s conceptual **Bubble/Depression gauge**:

1. **Data construction**
   - Macro data was extracted for **15 historical depressive periods** described in *Principles for Navigating Big Debt Crises*.
   - Each observation is labeled by **scenario**, not by year/country, so the model learns the direct relationship between economic features and the Bubble/Depression gauge.

2. **Models**
   - **Linear regression** on historical scenarios  
   - **Cubic spline interpolation** (piecewise cubic polynomial) to better capture non‑linearities in the gauge

3. **Application**
   - Both models are fit on the historical scenarios and then applied to **recent U.S. data (2017–2025)**.
   - Output: an implied **Bubble/Depression gauge over time** for the U.S.

4. **Finding**
   - Both models show a **downward trend from a peak around 2021**.  
   - The spline interpolation model roughly **doubles the R²** vs. the linear model (≈30% → ≈60%), indicating a better fit.

---

### 1.2 Depression Probability (Next‑Year Real GDP < 0)

The second part of the project asks:  
> *What is the probability that real GDP will contract next year (i.e., a “depression” event)?*

1. **Data**
   - Features: **14 macro variables** across developed countries from Karsten Müller’s **Global Macro Database**.
   - Target: **binary depression label for the following year**
     - `1` if **next year’s real GDP contracts**
     - `0` if **next year’s real GDP grows**

2. **Model**
   - **Random Fourier Features Logistic Regression with ridge regularization**
   - Trained and validated using **time‑series cross‑validation** to avoid look‑ahead bias and overfitting.

3. **Evaluation**
   - Initial training on 1875–1985 (Western nations), validation on 1985–2015.
   - Performance metrics:
     - **Brier Score** – measures calibration of probability estimates
     - **AUC (Area Under ROC Curve)** – measures ranking ability between depression and non‑depression years

4. **Findings**
   - Raw probability scores are **not perfectly calibrated** (Brier score is subpar), so raw probabilities should be interpreted with caution.
   - However, **AUC is high**, indicating that the model **ranks high‑risk vs. low‑risk years well**.
   - When applied to recent data, the model suggests:
     - A **heightened risk of depression next year** for the U.S.
     - Implied risk of real GDP contraction up to **~4× the historical average** (historical baseline ≈ 10%).

---

## 2. Data

> You will need **six core datasets** (plus two visualization spreadsheets) to run all analyses.

### 2.1 Ray Dalio’s Debt Cycle Predictor  
Directory: `DebtCycleGauge/Data/`

- **`Database2.csv`**  
  - Macro data and Bubble/Depression gauge values across **15 historical time periods** from *Big Debt Crises*.  
  - Labeled by **scenario**, not by year/country, to focus on the relationship between economic features and the gauge.

- **`PredictionBDG.csv`**  
  - U.S. macro data from **2017–2025**.  
  - The Debt Cycle models are applied to this file to generate **current Bubble/Depression gauge estimates**.

- **`DataVisualization.xlsx`**  
  - Optional visualization tool for exploring:
    - Model outputs
    - Underlying features  
  - Includes a **glossary of all features** used in the Debt Cycle model.

---

### 2.2 Depression Gauge  
Directory: `DepressionGauge/Database/`

- **`GMB.xlsx`**  
  - The full **Global Macro Database** with long‑run macro data across many countries (since 1086).

- **`TrainML.csv`**  
  - Training set: macro data for Western nations from **1875–1985** to train the initial ML model.

- **`ValidationML.csv`**  
  - Validation set: macro data for Western nations from **1985–2015** to evaluate out‑of‑sample performance.

- **`Train2ML.csv`**  
  - Combined training dataset: `TrainML.csv` + `ValidationML.csv`.  
  - Used for final model training on **all available historical data**.

- **`ForecastML3.csv`**  
  - Macro data for **25 countries** used to forecast **2026 depression probabilities**.

- **`PredictML2.csv`**  
  - Data for Western nations from **2015–2022**.  
  - Used to test the model’s ability to **anticipate the COVID‑19 contraction**.

- **`DataVisualization2.xlsx`**  
  - Optional visualization tool for the Depression Gauge models.  
  - Includes model outputs and a **glossary of all features** used.

---

## 3. Code & Workflow

### 3.1 DebtCycleGauge

Directory: `DebtCycleGauge/`

- **`DebtGauge.ipynb`**
  1. Runs a **linear regression** on `Database2.csv`:
     - Estimates coefficients linking macro features to the Bubble/Depression gauge.
     - Applies the model to `PredictionBDG.csv` to generate **current gauge values**.
  2. Runs a **piecewise cubic polynomial (cubic spline) interpolation** on `Database2.csv`:
     - Produces a non‑linear fit for the relationship between macro features and the gauge.
     - Re‑applies the resulting function to `PredictionBDG.csv`.

Both versions generate a time series of **implied Bubble/Depression gauge values**, showing a **decline from a 2021 peak** with the spline achieving a higher R².

---

### 3.2 DepressionGauge

Directory: `DepressionGauge/`

Scripts/notebooks in this directory implement three main pipelines:

1. **Initial Model & Validation**
   - Train **Random Fourier Features Logistic Regression (with ridge)** on `TrainML.csv`.
   - Evaluate on `ValidationML.csv` using:
     - **Brier score** (probability calibration)
     - **AUC** (ranking performance)

2. **COVID‑19 Backtest**
   - Retrain the model on `Train2ML.csv` (full historical data).  
   - Apply it to `PredictML2.csv` to:
     - Test how well it predicts the **COVID‑19 recession**.
     - Recompute Brier & AUC scores out‑of‑sample.

3. **2026 Forecast**
   - Use the model trained on `Train2ML.csv` to predict on `ForecastML3.csv`.  
   - Output: **raw probabilities of a depression in 2026** for **25 countries**.

Overall, the model shows:

- **Weak calibration** in predicting the COVID‑19 event (Brier score is not very strong).  
- **Strong ranking ability** (high AUC), indicating meaningful signal in separating high‑risk from low‑risk environments.  
- A **heightened probability of near‑term depression** for the U.S. relative to historical averages.

---

## 4. How to Run

1. **Clone the repo** and ensure the directory structure matches what is described above.
2. Place all datasets in the appropriate folders:
   - `DebtCycleGauge/Data/`
   - `DepressionGauge/Database/`
3. Open the notebooks/scripts in your Python environment (e.g., Jupyter, VS Code) and:
   - Run `DebtCycleGauge/DebtGauge.ipynb` to:
     - Recreate the Bubble/Depression gauge estimates for the U.S.
   - Run the scripts/notebooks in `DepressionGauge/` to:
     - Train the depression classifier
     - Validate historical performance
     - Generate forecasts (COVID and 2026)

Dependencies are standard for a machine‑learning macro project (Python, NumPy, pandas, scikit‑learn, etc.). See the notebooks for exact imports.

---

## 5. Glossary of Concepts

A quick reference for the main methods used in this project.

- **Machine Learning**  
  A broad term for using algorithms to learn patterns from data and **evaluate them on out‑of‑sample data** to see how well they generalize.  
  Here, we use:
  - Linear regression  
  - Non‑linear regression via **cubic spline interpolation**  
  - **Random Fourier Features Logistic Regression** with ridge regularization

- **Cubic Spline Interpolation**  
  Instead of fitting a single line \( Y = mX + b \), cubic splines fit **piecewise cubic polynomials** joined smoothly at “knots”.  
  This allows the model to capture **non‑linear relationships** between predictors (X) and the target (Y).

- **Random Fourier Features (RFF)**  
  A technique to approximate certain **kernel methods** by projecting the data into a higher‑dimensional space using random cosine/sine features.  
  In practice:
  - Generate random weights from a distribution related to a chosen kernel.
  - Transform original features using these cosine‑based mappings.
  - Fit a **linear model** in this transformed space to approximate **non‑linear behavior** at lower computational cost than full kernel methods.

- **Ridge Regression**  
  A regularized regression that adds a **penalty on coefficient magnitude** to the loss function.  
  This helps:
  - Reduce **overfitting**
  - Stabilize models with correlated predictors  
  It is used here inside the logistic regression to regularize the Random Fourier features.

- **Logistic Regression**  
  A model used when the target is **binary** (e.g., depression vs. no depression).  
  It outputs **probabilities** between 0 and 1 for the positive class (here, “depression next year”).

- **Brier Score**  
  A measure of how well predicted probabilities match actual outcomes.  
  - Lower = better calibration  
  - A poor Brier score means the **raw probabilities** should be treated cautiously.

- **AUC (Area Under the ROC Curve)**  
  Measures how well the model **ranks** positive cases above negative ones, independent of any probability cutoff.  
  - Random guessing → AUC ≈ 0.5  
  - Higher AUC → better separation between depression and non‑depression years  
  This project relies heavily on AUC to judge whether the model contains **useful ranking information**, even if probabilities are imperfectly calibrated.

---

## Citation

If you use this code or data, please cite:

> Mikkel Shiffman (2025). *Machine Learning on the U.S. Debt Cycle*.

