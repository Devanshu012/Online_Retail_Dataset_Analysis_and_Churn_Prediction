# 📊 E-Commerce Customer Analytics Dashboard

> An interactive, multi-tab analytics dashboard that transforms raw e-commerce transaction data into actionable business insights — from customer geography and spending behaviour to RFM segmentation, churn analysis, and ML-powered churn prediction.

---

## 🖼️ Preview

> _Add screenshots of each tab here once the app is running._
> ```
> ![Dashboard Preview](assets/preview.png)
> ```

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 📖 About the Project

This project started as a Jupyter Notebook exploration of an online retail dataset and evolved into a fully interactive Streamlit dashboard. It covers the complete analytics pipeline:

- **Descriptive analytics** — who are the customers, where are they, how much do they spend?
- **Behavioural segmentation** — RFM (Recency, Frequency, Monetary) analysis to classify customer loyalty
- **Churn analytics** — identifying customers who have gone quiet and understanding patterns
- **Predictive modelling** — a Logistic Regression model to predict which customers are likely to churn

Everything runs in a single `streamlit run app.py` command with no extra setup beyond installing the requirements.

---

## ✨ Features

### 🌍 Customers by Country
- Top 10 countries ranked by unique customer count
- Percentage share displayed on each bar
- Interactive hover with exact figures

### 🏆 Top Customers by Visits
- Ranks customers by number of invoice visits
- Each bar labelled with Customer ID and country of origin

### 💸 Top Spenders
- Top 10 customers by total revenue contribution
- Spend values displayed in ₹ on chart and hover

### 📈 Monthly Sales Trend
- Line chart of total monthly revenue over time
- Surfaces seasonality, growth spurts, and anomalies

### 🎯 RFM Analysis
- Computes Recency, Frequency, and Monetary scores per customer
- Assigns customers to 6 segments: **Champions, Loyal, Recent, At Risk, Lost, Others**
- Segment distribution bar chart and KPI cards
- R/F/M histogram distributions (side-by-side)
- Interactive RFM heatmap (avg monetary value by quartile)
- Top Champions table (RFM score = 111)

### 🔄 Churn Analysis
- Defines churn as no purchase in the last **90 days**
- KPI cards: total customers, churned count & %, retained count & %
- Stacked bar chart of churned vs retained by country
- Toggle to exclude United Kingdom (which dominates volume)
- Churn rate % bar chart — top 15 countries by volume
- Overlapping histogram of days since last purchase (churned vs retained)

### 🤖 Churn Prediction Model
- Logistic Regression trained on RFM + transactional features
- **PCA scatter plot** — 2D visualisation of customer separation
- **ROC Curve** with AUC score vs random baseline
- **Confusion Matrix** heatmap
- **Feature Coefficient** chart — which features drive churn risk
- **Classification Report** table — precision, recall, F1 per class
- Model cached with `@st.cache_data` — trains once per session

---

## 🛠️ Tech Stack

| Purpose | Library |
|---|---|
| Dashboard & UI | `streamlit` |
| Interactive charts | `plotly`, `plotly.express`, `plotly.graph_objects` |
| Data wrangling | `pandas`, `numpy` |
| Machine learning | `scikit-learn` |
| Dimensionality reduction | `sklearn.decomposition.PCA` |
| Statistical plots (notebook) | `matplotlib`, `seaborn` |

---

## 📁 Project Structure

```
ecommerce-analytics-dashboard/
│
├── app.py                          # Main Streamlit application
│
├── Plots/
│   └── plotter.py                  # Reusable Plotly chart methods (Plotter class)
│
├── Data/
│   └── cleaned_customer_with_rfm.csv   # Preprocessed dataset with RFM columns
│
├── notebooks/
│   └── notebook.ipynb              # EDA, RFM, churn analysis & model prototyping
│
├── requirements.txt                # Python dependencies
└── README.md                       # You are here
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ecommerce-analytics-dashboard.git
cd ecommerce-analytics-dashboard

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### requirements.txt

```
streamlit
pandas
numpy
plotly
scikit-learn
seaborn
matplotlib
```

---

## 📦 Dataset

The project uses the **[UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail)** — transactional records from a UK-based online retailer covering **December 2010 to December 2011**.

| Column | Description |
|---|---|
| `InvoiceNo` | Unique invoice identifier |
| `StockCode` | Product code |
| `Description` | Product name |
| `Quantity` | Units purchased |
| `InvoiceDate` | Date and time of transaction |
| `UnitPrice` | Price per unit |
| `CustomerID` | Unique customer identifier |
| `Country` | Customer's country |

The CSV in `/Data` is a cleaned version with additional engineered columns:

| Added Column | Description |
|---|---|
| `TotalPrice` | `Quantity × UnitPrice` |
| `Year`, `Month`, `Day` | Extracted from `InvoiceDate` |
| `recency` | Days since customer's last purchase |
| `frequency` | Number of invoices |
| `monetary_value` | Total spend |

---

## 🔍 How It Works

### RFM Scoring
Each customer receives a quartile score (1–4) for Recency, Frequency, and Monetary value. The scores are combined into a 3-digit `RFMScore`. Customers with score `111` are **Champions** — most recent, most frequent, highest spenders.

### Churn Definition
A customer is marked as **churned** if their last purchase was more than **90 days** before the dataset's most recent transaction date (`2011-12-09`).

### Churn Prediction Features
The model is trained on:
- `UK_Flag` — whether the customer is based in the UK
- `TotalPrice` — transaction-level spend
- `recency`, `frequency`, `monetary_value` — RFM features

Features are standardised with `StandardScaler` before fitting Logistic Regression. The model achieves **~99.97% accuracy** on this dataset due to the strong signal in recency.

---

## 🔮 Future Improvements

- [ ] Add a **customer lookup** tool — enter a Customer ID and get their full profile + churn probability
- [ ] Try ensemble models (Random Forest, XGBoost) for comparison
- [ ] Add **date range filters** to all charts via sidebar
- [ ] Deploy to **Streamlit Cloud** for public access
- [ ] Add **product-level analysis** — best sellers, return rates, category breakdown
- [ ] Export filtered data and charts as PDF/Excel

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

Built by **Devanshu Singh(https://github.com/Devanshu012)**  
Feel free to open an issue or submit a pull request!

---

> ⭐ If you found this useful, consider giving the repo a star!
