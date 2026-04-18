import plotly.express as px
import pandas as pd

class Plotter:

    def customers_by_country(self, df):
        customers_per_country = (
            df.groupby('Country')['CustomerID']
            .nunique()
            .reset_index(name='Customer_Count')
            .sort_values(by='Customer_Count', ascending=False)
        )

        total = customers_per_country['Customer_Count'].sum()
        customers_per_country['Percentage'] = (
            customers_per_country['Customer_Count'] / total * 100
        )

        top = customers_per_country.head(10)

        fig = px.bar(
            top,
            x='Country',
            y='Customer_Count',
            text=top.apply(
                lambda x: f"{x['Customer_Count']} ({x['Percentage']:.1f}%)",
                axis=1
            ),
            color='Customer_Count',
            color_continuous_scale='Blues',
            title="Top 10 Countries by Customer Count"
        )

        fig.update_traces(
            textposition='outside',
            customdata=top['Percentage'],
            hovertemplate="<b>%{x}</b><br>Customers: %{y}<br>Percentage: %{customdata:.2f}%"
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            template="plotly_white",
            height=500
        )

        return fig


    def top_customers(self, df):
        visits = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index(name='Visit_Count')
        country = df.groupby('CustomerID')['Country'].first().reset_index()

        data = visits.merge(country, on='CustomerID')
        top = data.sort_values(by='Visit_Count', ascending=False).head(10)

        top['Label'] = top['CustomerID'].astype(str) + " (" + top['Country'] + ")"

        fig = px.bar(
            top,
            x='Label',
            y='Visit_Count',
            text='Visit_Count',
            color='Visit_Count',
            color_continuous_scale='Viridis',
            title="Top 10 Customers by Number of Visits"
        )

        fig.update_traces(
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Visits: %{y}"
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            template="plotly_white",
            height=500
        )

        return fig
    
    def top_spenders(self, df):
        spend = df.groupby('CustomerID')['TotalPrice'].sum().reset_index()
        country = df.groupby('CustomerID')['Country'].first().reset_index()

        data = spend.merge(country, on='CustomerID')
        top = data.sort_values(by='TotalPrice', ascending=False).head(10)

        # Create label
        top['Label'] = top['CustomerID'].astype(str) + " (" + top['Country'] + ")"

        fig = px.bar(
            top,
            x='Label',
            y='TotalPrice',
            text=top['TotalPrice'].apply(lambda x: f"₹{x:,.0f}"),
            color='TotalPrice',
            color_continuous_scale='Plasma',
            title="Top 10 Customers by Total Spending"
        )

        fig.update_traces(
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Total Spend: ₹%{y:,.2f}"
        )

        fig.update_layout(
            xaxis_title="Customer (Country)",
            yaxis_title="Total Spend (₹)",
            xaxis_tickangle=-45,
            template="plotly_white",
            height=500
        )

        return fig, top
    
    def monthly_sales(self, df):
        monthly = (
            df[df['Quantity'] > 0]
            .groupby(['Year', 'Month'])['TotalPrice']
            .sum()
            .reset_index()
        )

        # Create proper datetime column
        monthly['Date'] = pd.to_datetime(
            monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str)
        )

        monthly = monthly.sort_values('Date')

        fig = px.line(
            monthly,
            x='Date',
            y='TotalPrice',
            markers=True,
            title="Monthly Sales Over Time"
        )

        fig.update_traces(
            hovertemplate="<b>%{x|%b %Y}</b><br>Sales: ₹%{y:,.2f}"
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Total Sales (₹)",
            template="plotly_white",
            height=500
        )

        return fig, monthly