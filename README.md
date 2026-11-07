# 🛒 E-commerce Customer Spending Predictor

A machine learning project that predicts a customer's **Yearly Amount Spent** from their app/website behaviour and membership history. It includes an end-to-end workflow (EDA → feature engineering → model building → tuning) and a **Streamlit web app** for real-time predictions.

---

## 📌 Problem Statement

An online clothing retailer runs both a mobile app and a website. The business wants to understand how customer engagement drives revenue and to estimate how much a customer is likely to spend in a year, so it can decide where to focus effort (app vs. website) and which customers to retain.

**Goal:** Build a regression model that predicts `Yearly Amount Spent` for a customer.

---

## 📂 Project Structure

```
├── app.py            # Streamlit web app for predictions
├── NOTEBOOK.ipynb    # EDA, feature engineering, modelling & tuning
├── data.csv          # Dataset (500 customers)
├── models/
│   ├── model.pkl     # Trained regression model
│   └── scaler.pkl    # Fitted StandardScaler
└── README.md
```

> The notebook saves `model.pkl` and `scaler.pkl` in its working directory. Move them into a `models/` folder, which is where `app.py` loads them from.

---

## 📊 Dataset

`data.csv` has **500 rows and 8 columns**.

| Column | Description |
|---|---|
| `Email` | Customer email (dropped – identifier) |
| `Address` | Customer address (dropped – identifier) |
| `Avatar` | Avatar colour (dropped – not informative) |
| `Avg. Session Length` | Average length of in-store style advice sessions (minutes) |
| `Time on App` | Average time spent on the mobile app (minutes) |
| `Time on Website` | Average time spent on the website (minutes) |
| `Length of Membership` | Number of years the customer has been a member |
| `Yearly Amount Spent` | **Target** – amount spent by the customer per year |

---

## 🔬 Workflow

1. **Data understanding** – shape, null values, duplicates, `info()` and `describe()`.
2. **Cleaning** – dropped the `Email`, `Address` and `Avatar` columns (unique/identifier-like).
3. **EDA**
   - Univariate: histograms and skewness for every numeric feature.
   - Multivariate: pair plot.
   - Outlier detection with box plots.
4. **Feature engineering**
   - Removed outliers using the **IQR method** (24 rows removed).
   - Checked **multicollinearity** with VIF (all features ≈ 1, so no multicollinearity).
   - **Feature selection** with correlation analysis and **RFE**. The selected features were `Avg. Session Length`, `Time on App` and `Length of Membership`; `Time on Website` was left out.
5. **Modelling**
   - 80/20 train-test split (`random_state=42`) and `StandardScaler` on the features.
   - Compared Linear Regression, Lasso, Ridge and ElasticNet.
   - Hyperparameter tuning of ElasticNet with `GridSearchCV` (5-fold CV, scored on negative MAE).
6. **Deployment** – saved the model and scaler with `pickle` and built a Streamlit UI.

---

## 📈 Results

Baseline comparison on the test set (scaled features):

| Model | MSE | R² | MAE |
|---|---|---|---|
| Linear Regression | 104.97 | 0.98 | 8.07 |
| Lasso (alpha=1) | 106.20 | 0.98 | 8.25 |
| Ridge (alpha=1) | 104.92 | 0.98 | 8.08 |
| ElasticNet (alpha=1) | 739.15 | 0.85 | 22.21 |

Tuned ElasticNet (`GridSearchCV`): best params `alpha=0.1`, `l1_ratio=0.9`, with a cross-validated MAE of about **8.08**.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Poorva2004/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install streamlit numpy pandas scikit-learn matplotlib seaborn statsmodels jupyter
```

Or create a `requirements.txt`:

```
streamlit
numpy
pandas
scikit-learn
matplotlib
seaborn
statsmodels
```

### 3. (Optional) Re-train the model

Open `NOTEBOOK.ipynb`, run all cells, and place the generated `model.pkl` and `scaler.pkl` inside a `models/` folder.

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🖥️ Using the App

Enter the three inputs and click **Predict**:

| Input | Example |
|---|---|
| Average Session Length | 35 |
| Time on App | 10 |
| Length of Membership | 5 |

The app scales the inputs with the saved scaler and shows the predicted **Yearly Amount Spent**.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data & ML:** Pandas, NumPy, scikit-learn, statsmodels
- **Visualisation:** Matplotlib, Seaborn
- **App:** Streamlit
- **Tools:** Jupyter Notebook, Git & GitHub

---

## 🔮 Future Improvements

- Add a held-out evaluation of the final tuned model (using scaled test data) and report it in this README.
- Try tree-based models (Random Forest, XGBoost) and compare.
- Add input validation and a feature-importance chart in the app.
- Deploy on Streamlit Community Cloud.

---

## 👩‍💻 Author

**Poorva Chavan**
- GitHub: [Poorva2004](https://github.com/Poorva2004)
- LinkedIn: [poorvachavan](https://linkedin.com/in/poorvachavan)
