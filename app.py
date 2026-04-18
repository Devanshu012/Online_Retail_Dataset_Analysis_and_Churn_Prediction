import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from Plots.plotter import Plotter


st.set_page_config(layout="wide")

st.title("📊 E-Commerce Data Insights Dashboard")


# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("Data/cleaned_customer_with_rfm.csv")


@st.cache_data
def compute_rfm(df):
    NOW = dt.datetime(2011, 12, 10)

    df1 = df.copy()
    df1['InvoiceDate'] = pd.to_datetime(df1['InvoiceDate'])

    rfmTable = df1.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (NOW - x.max()).days,
        'InvoiceNo': lambda x: len(x),
        'TotalPrice': lambda x: x.sum()
    })

    rfmTable.rename(columns={
        'InvoiceDate': 'recency',
        'InvoiceNo': 'frequency',
        'TotalPrice': 'monetary_value'
    }, inplace=True)

    quantiles = rfmTable.quantile(q=[0.25, 0.5, 0.75]).to_dict()

    segmented_rfm = rfmTable.copy()

    def RScore(x, p, d):
        if x <= d[p][0.25]:   return 1
        elif x <= d[p][0.50]: return 2
        elif x <= d[p][0.75]: return 3
        else:                  return 4

    def FMScore(x, p, d):
        if x <= d[p][0.25]:   return 4
        elif x <= d[p][0.50]: return 3
        elif x <= d[p][0.75]: return 2
        else:                  return 1

    segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(RScore, args=('recency', quantiles,))
    segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency', quantiles,))
    segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(FMScore, args=('monetary_value', quantiles,))

    segmented_rfm['RFMScore'] = (
        segmented_rfm.r_quartile.map(str) +
        segmented_rfm.f_quartile.map(str) +
        segmented_rfm.m_quartile.map(str)
    )

    def segment_customer(row):
        if row['RFMScore'] == '111':
            return 'Champions'
        elif row['r_quartile'] <= 2 and row['f_quartile'] <= 2:
            return 'Loyal Customers'
        elif row['r_quartile'] == 1:
            return 'Recent Customers'
        elif row['r_quartile'] >= 3 and row['f_quartile'] <= 2:
            return 'At Risk'
        elif row['r_quartile'] == 4:
            return 'Lost Customers'
        else:
            return 'Others'

    segmented_rfm['Segment'] = segmented_rfm.apply(segment_customer, axis=1)

    return segmented_rfm


df = load_data()
plotter = Plotter()

# Sidebar
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Select Insight",
    [
        "Customers by Country",
        "Top Customers",
        "Top Spenders",
        "Monthly Sales",
        "RFM Analysis",
        "Churn Analysis",
        "Churn Prediction Model",   # ← NEW
    ]
)

# ── existing tabs ────────────────────────────────────────────────────────────
if option == "Customers by Country":
    st.header("🌍 Customers per Country")
    fig = plotter.customers_by_country(df)
    st.plotly_chart(fig, use_container_width=True)

elif option == "Top Customers":
    st.header("🏆 Top Customers by Visits")
    fig = plotter.top_customers(df)
    st.plotly_chart(fig, use_container_width=True)

elif option == "Top Spenders":
    st.header("🏆 Top Spenders")
    fig, top = plotter.top_spenders(df)
    st.plotly_chart(fig, use_container_width=True)

elif option == "Monthly Sales":
    st.header("📈 Monthly Sales Trend")
    fig, monthly = plotter.monthly_sales(df)
    st.plotly_chart(fig, use_container_width=True)

# ── NEW: RFM Analysis tab ────────────────────────────────────────────────────
elif option == "RFM Analysis":
    st.header("🎯 RFM Customer Segmentation")

    segmented_rfm = compute_rfm(df)

    # ── KPI row ──────────────────────────────────────────────────────────────
    segments = segmented_rfm['Segment'].value_counts()
    cols = st.columns(len(segments))
    icons = {
        'Champions': '🏆',
        'Loyal Customers': '💛',
        'Recent Customers': '🆕',
        'At Risk': '⚠️',
        'Lost Customers': '💀',
        'Others': '👥',
    }
    for col, (seg, count) in zip(cols, segments.items()):
        col.metric(f"{icons.get(seg, '')} {seg}", count)

    st.divider()

    # ── Row 1: Segment bar + RFM distributions ───────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Segment Distribution")
        fig_seg = px.bar(
            x=segments.index,
            y=segments.values,
            color=segments.index,
            labels={'x': 'Segment', 'y': 'Customer Count'},
            color_discrete_sequence=px.colors.qualitative.Bold,
            template='plotly_white',
        )
        fig_seg.update_layout(
            showlegend=False,
            xaxis_tickangle=-30,
            height=400,
        )
        fig_seg.update_traces(
            texttemplate='%{y}',
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Customers: %{y}",
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with col2:
        st.subheader("R / F / M Distributions")
        fig_dist = make_subplots(rows=1, cols=3, subplot_titles=("Recency", "Frequency", "Monetary Value"))

        for i, col_name in enumerate(['recency', 'frequency', 'monetary_value'], start=1):
            fig_dist.add_trace(
                go.Histogram(
                    x=segmented_rfm[col_name],
                    name=col_name.capitalize(),
                    marker_color=['#636EFA', '#EF553B', '#00CC96'][i - 1],
                    opacity=0.8,
                ),
                row=1, col=i
            )

        fig_dist.update_layout(
            template='plotly_white',
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # ── Row 2: RFM Heatmap ───────────────────────────────────────────────────
    st.subheader("RFM Heatmap — Average Monetary Value by Recency × Frequency Quartile")

    pivot = segmented_rfm.pivot_table(
        index='r_quartile',
        columns='f_quartile',
        values='monetary_value',
        aggfunc='mean'
    )

    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"F{c}" for c in pivot.columns],
        y=[f"R{r}" for r in pivot.index],
        colorscale='RdYlGn',
        text=np.round(pivot.values, 0).astype(int),
        texttemplate="%{text}",
        hovertemplate="Recency: %{y}<br>Frequency: %{x}<br>Avg Monetary: ₹%{z:,.0f}",
    ))

    fig_heat.update_layout(
        xaxis_title="Frequency Quartile",
        yaxis_title="Recency Quartile",
        template='plotly_white',
        height=380,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # ── Row 3: Champions table ───────────────────────────────────────────────
    st.subheader("🏆 Top Champions (RFM Score = 111)")
    champions = (
        segmented_rfm[segmented_rfm['RFMScore'] == '111']
        .sort_values('monetary_value', ascending=False)
        .head(10)
        .reset_index()
    )
    champions['monetary_value'] = champions['monetary_value'].map('₹{:,.2f}'.format)
    st.dataframe(champions, use_container_width=True)


# ── NEW: Churn Analysis tab ──────────────────────────────────────────────────
elif option == "Churn Analysis":
    st.header("🔄 Customer Churn Analysis")

    @st.cache_data
    def compute_churn(df):
        d = df.copy()
        d['InvoiceDate'] = pd.to_datetime(d['InvoiceDate'])
        max_date = d['InvoiceDate'].max()
        d['LastPurchaseDate'] = d.groupby('CustomerID')['InvoiceDate'].transform('max')
        d['DaysSinceLastPurchase'] = (max_date - d['LastPurchaseDate']).dt.days
        d['Churn'] = (d['DaysSinceLastPurchase'] > 90).astype(int)

        customer_data = d[['CustomerID', 'Country', 'Churn']].drop_duplicates()
        churn_country = (
            customer_data
            .groupby(['Country', 'Churn'])['CustomerID']
            .nunique()
            .reset_index()
        )
        churn_pivot = (
            churn_country
            .pivot(index='Country', columns='Churn', values='CustomerID')
            .fillna(0)
            .rename(columns={0: 'Not_Churned', 1: 'Churned'})
            .sort_values(by='Churned', ascending=False)
        )
        churned_count  = int(d[d['Churn'] == 1]['CustomerID'].nunique())
        retained_count = int(d[d['Churn'] == 0]['CustomerID'].nunique())
        return d, churn_pivot, churned_count, retained_count

    raw, churn_pivot, churned_count, retained_count = compute_churn(df)

    # ── KPI row ──────────────────────────────────────────────────────────────
    total = churned_count + retained_count
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Unique Customers", total)
    k2.metric("🔴 Churned  (>90 days)", churned_count,
              delta=f"{churned_count/total*100:.1f}% of total", delta_color="inverse")
    k3.metric("🟢 Retained (≤90 days)", retained_count,
              delta=f"{retained_count/total*100:.1f}% of total")

    st.divider()

    # ── Toggle: include / exclude UK ─────────────────────────────────────────
    exclude_uk = st.toggle("Exclude United Kingdom (dominates chart)", value=False)
    plot_data = churn_pivot.drop('United Kingdom', errors='ignore') if exclude_uk else churn_pivot

    # ── Stacked bar chart ────────────────────────────────────────────────────
    st.subheader(
        "Customer Churn by Country" +
        (" (Excluding UK)" if exclude_uk else "")
    )

    fig_churn = go.Figure()
    fig_churn.add_trace(go.Bar(
        name='Not Churned',
        x=plot_data.index,
        y=plot_data['Not_Churned'],
        marker_color='#2ecc71',
        hovertemplate="<b>%{x}</b><br>Not Churned: %{y}",
    ))
    fig_churn.add_trace(go.Bar(
        name='Churned',
        x=plot_data.index,
        y=plot_data['Churned'],
        marker_color='#e74c3c',
        hovertemplate="<b>%{x}</b><br>Churned: %{y}",
    ))
    fig_churn.update_layout(
        barmode='stack',
        xaxis_tickangle=-45,
        xaxis_title='Country',
        yaxis_title='Number of Customers',
        template='plotly_white',
        height=500,
        legend_title='Churn Status',
    )
    st.plotly_chart(fig_churn, use_container_width=True)

    st.divider()

    # ── Churn rate per country (top 15) ──────────────────────────────────────
    st.subheader("Churn Rate % by Country (Top 15 by volume)")

    rate_df = churn_pivot.copy()
    rate_df['Total'] = rate_df['Not_Churned'] + rate_df['Churned']
    rate_df['Churn_Rate'] = rate_df['Churned'] / rate_df['Total'] * 100
    rate_df = rate_df.sort_values('Total', ascending=False).head(15)

    fig_rate = px.bar(
        rate_df.reset_index(),
        x='Country',
        y='Churn_Rate',
        color='Churn_Rate',
        color_continuous_scale='RdYlGn_r',
        text=rate_df['Churn_Rate'].map('{:.1f}%'.format).values,
        template='plotly_white',
        height=420,
    )
    fig_rate.update_traces(
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.1f}%",
    )
    fig_rate.update_layout(
        xaxis_tickangle=-35,
        yaxis_title='Churn Rate (%)',
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_rate, use_container_width=True)

    st.divider()

    # ── Days-since-last-purchase distribution ────────────────────────────────
    st.subheader("Days Since Last Purchase — Churned vs Retained")

    customer_level = raw[['CustomerID', 'DaysSinceLastPurchase', 'Churn']].drop_duplicates()

    fig_hist = go.Figure()
    for label, color, val in [('Retained', '#2ecc71', 0), ('Churned', '#e74c3c', 1)]:
        subset = customer_level[customer_level['Churn'] == val]['DaysSinceLastPurchase']
        fig_hist.add_trace(go.Histogram(
            x=subset,
            name=label,
            marker_color=color,
            opacity=0.7,
            nbinsx=40,
        ))
    fig_hist.update_layout(
        barmode='overlay',
        xaxis_title='Days Since Last Purchase',
        yaxis_title='Number of Customers',
        template='plotly_white',
        height=400,
        legend_title='Status',
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ── NEW: Churn Prediction Model tab ─────────────────────────────────────────
elif option == "Churn Prediction Model":
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, classification_report,
        confusion_matrix, roc_curve, auc,
    )

    st.header("🤖 Churn Prediction — Logistic Regression")

    @st.cache_data
    def build_model(df):
        d = df.copy()
        d['InvoiceDate'] = pd.to_datetime(d['InvoiceDate'])
        max_date = d['InvoiceDate'].max()
        d['LastPurchaseDate'] = d.groupby('CustomerID')['InvoiceDate'].transform('max')
        d['DaysSinceLastPurchase'] = (max_date - d['LastPurchaseDate']).dt.days
        d['Churn'] = (d['DaysSinceLastPurchase'] > 90).astype(int)
        d['UK_Flag'] = (d['Country'] == 'United Kingdom').astype(int)

        cols = ['UK_Flag', 'TotalPrice', 'Year', 'Month', 'Day',
                'recency', 'frequency', 'monetary_value', 'Churn']
        df_new = d[cols].dropna()

        # PCA (visualisation only)
        df_reduced = df_new.drop(columns=['Year', 'Month', 'Day', 'Churn'])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(df_reduced)
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['Churn'] = df_new['Churn'].values

        # Model
        X = df_new.drop(columns=['Churn', 'Year', 'Month', 'Day'])
        y = df_new['Churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc    = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm     = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        feature_names = X.columns.tolist()
        coefs = model.coef_[0]

        return pca_df, acc, report, cm, fpr, tpr, roc_auc, feature_names, coefs

    with st.spinner("Training model..."):
        pca_df, acc, report, cm, fpr, tpr, roc_auc, feature_names, coefs = build_model(df)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy",             f"{acc*100:.2f}%")
    k2.metric("AUC-ROC",              f"{roc_auc:.4f}")
    k3.metric("Precision (churn)",    f"{report['1']['precision']*100:.1f}%")
    k4.metric("Recall (churn)",       f"{report['1']['recall']*100:.1f}%")

    st.divider()

    # Row 1: PCA scatter + ROC curve
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2D PCA — Customer Segmentation")
        fig_pca = px.scatter(
            pca_df,
            x='PC1', y='PC2',
            color=pca_df['Churn'].map({0: 'Retained', 1: 'Churned'}),
            color_discrete_map={'Retained': '#2ecc71', 'Churned': '#e74c3c'},
            opacity=0.55,
            template='plotly_white',
            height=420,
            labels={'color': 'Status'},
        )
        fig_pca.update_traces(marker_size=4)
        fig_pca.update_layout(legend_title_text='Status')
        st.plotly_chart(fig_pca, use_container_width=True)

    with col2:
        st.subheader(f"ROC Curve  (AUC = {roc_auc:.4f})")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'Logistic Regression (AUC={roc_auc:.4f})',
            line=dict(color='#636EFA', width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='grey', dash='dash'),
        ))
        fig_roc.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template='plotly_white',
            height=420,
            legend=dict(x=0.4, y=0.1),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.divider()

    # Row 2: Confusion matrix + Feature coefficients
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Confusion Matrix")
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=['Predicted: Retained', 'Predicted: Churned'],
            y=['Actual: Retained', 'Actual: Churned'],
            color_continuous_scale='Blues',
            template='plotly_white',
            height=380,
        )
        fig_cm.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col4:
        st.subheader("Feature Coefficients (Log-Odds)")
        coef_df = (
            pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
            .sort_values('Coefficient')
        )
        colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in coef_df['Coefficient']]
        fig_coef = go.Figure(go.Bar(
            x=coef_df['Coefficient'],
            y=coef_df['Feature'],
            orientation='h',
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Coefficient: %{x:.4f}",
        ))
        fig_coef.update_layout(
            xaxis_title='Coefficient (log-odds)',
            template='plotly_white',
            height=380,
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    st.divider()

    # Classification report table
    st.subheader("Classification Report")
    report_df = pd.DataFrame(report).T.drop(index='accuracy', errors='ignore')
    report_df = report_df[['precision', 'recall', 'f1-score', 'support']].round(4)
    report_df.index = report_df.index.map(
        lambda x: {'0': 'Retained', '1': 'Churned'}.get(x, x)
    )
    st.dataframe(report_df, use_container_width=True)