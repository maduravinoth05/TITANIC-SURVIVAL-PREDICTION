# TITANIC-SURVIVAL-PREDICTION
# 🚢 Titanic Survival Prediction using Artificial Intelligence
CodTech IT Solutions — Artificial intelligence Internship  
Task name : TITANIC SURVIVAL PREDICTION    
Intern : VINOTH V  
Intern ID : CITS7154  
Domain : Artificial Intelligence    
Duration : 4 Weeks  
Internship Period : 16 JULY 2026 - 13 AUGUST 2026  
---

## 📌 Project Overview

This project builds a **Machine Learning classification model** that predicts
whether a passenger aboard the RMS Titanic would have **survived or not**,
based on attributes such as age, sex, ticket class, fare paid, and family
size. It follows the classic **Titanic: Machine Learning from Disaster**
dataset structure popularized by Kaggle.

The goal of this internship task is to demonstrate the complete AI/ML
workflow:

1. Data loading & exploration
2. Data cleaning & preprocessing (handling missing values, encoding)
3. Feature engineering
4. Model training (Logistic Regression & Random Forest)
5. Model evaluation (accuracy, precision, recall, F1-score, confusion matrix)
6. Feature importance analysis
7. Making predictions on new/unseen passenger data

---

## 🧠 Algorithms Used

| Model | Type | Purpose |
|---|---|---|
| Logistic Regression | Linear classifier | Baseline model |
| Random Forest Classifier | Ensemble tree-based model | Higher accuracy + feature importance |

---

## 🗂️ Project Structure

```
titanic-survival-prediction/
│
├── titanic_survival_prediction.py   # Main program (data → model → results)
├── train.csv                        # Titanic dataset (Kaggle) - add your own
├── sample_output.txt                # Sample console output of a run
├── terminal_screenshot.png          # Screenshot of the program running
├── README.md                        # Project documentation (this file)
└── requirements.txt                 # Python dependencies
```

---

## 📊 Dataset

This project uses the standard **Kaggle Titanic dataset** (`train.csv`),
containing 891 passenger records with the following columns:

| Column | Description |
|---|---|
| PassengerId | Unique passenger ID |
| Survived | Target — 0 = Died, 1 = Survived |
| Pclass | Ticket class (1st, 2nd, 3rd) |
| Name, Sex, Age | Passenger demographics |
| SibSp, Parch | Siblings/spouses & parents/children aboard |
| Ticket, Fare | Ticket number and fare paid |
| Cabin, Embarked | Cabin number & port of embarkation |

📥 **Download it here:** [Kaggle – Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)
Place the downloaded `train.csv` in the project root folder.

> **Note:** If `train.csv` is not found, the script automatically generates
> a *synthetic* Titanic-like dataset (same columns/schema) so the project
> still runs end-to-end for demonstration/testing purposes.

---

## ⚙️ Installation & Setup

```bash
# 1. Clone this repository
git clone https://github.com/<your-username>/titanic-survival-prediction.git
cd titanic-survival-prediction

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`
```
pandas
numpy
scikit-learn
```

---

## ▶️ How to Run

```bash
python titanic_survival_prediction.py
```

The script will:
- Load and explore the dataset
- Clean & preprocess the data
- Train Logistic Regression and Random Forest models
- Print accuracy, classification report & confusion matrix for each model
- Display feature importance
- Predict the outcome for a sample passenger

---


## 🔍 Key Insights

- **Sex** and **Passenger Class** are among the strongest predictors of
  survival — reflecting the historical "women and children first" policy
  and better lifeboat access for higher-class passengers.
- **Fare** correlates strongly with class and cabin location, also
  influencing survival chances.
- The **Random Forest** model generally outperforms plain Logistic
  Regression due to its ability to capture non-linear feature interactions.

---

## 🚀 Future Improvements

- Hyperparameter tuning with `GridSearchCV`
- Add more models (SVM, XGBoost, Gradient Boosting)
- Cross-validation for more robust accuracy estimates
- Deploy as a web app (Flask/Streamlit) for interactive predictions

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, NumPy, scikit-learn
- **Tools:** VS Code / Jupyter Notebook, Git & GitHub

---

