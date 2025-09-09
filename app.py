import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import joblib

# Load the trained model and the original data
model = joblib.load('churn_model.pkl')
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# --- Initialize the Dash App ---
app = dash.Dash(__name__)
server = app.server # Expose server for deployment

# --- Prepare Data for Visualizations ---
# KPI 1: Churn Rate
total_customers = len(df)
churned_customers = df['Churn'].value_counts().get('Yes', 0)
churn_rate = (churned_customers / total_customers) * 100

# Chart 1: Churn by Contract Type
churn_by_contract = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack().fillna(0)
fig_contract = px.bar(churn_by_contract, x=churn_by_contract.index, y='Yes',
                      title='Churn Rate by Contract Type', labels={'x': 'Contract Type', 'Yes': 'Churn Rate'},
                      text_auto='.2%', color_discrete_sequence=['#007bff'])

# Chart 2: Churn by Tenure
fig_tenure = px.histogram(df, x='tenure', color='Churn',
                          title='Customer Tenure Distribution by Churn Status',
                          labels={'tenure': 'Tenure (Months)'},
                          color_discrete_map={'Yes': '#dc3545', 'No': '#28a745'})

# --- Define the Dashboard Layout ---
app.layout = html.Div(className='container', children=[
    html.H1("Customer Churn Prediction Dashboard"),
    html.Hr(),

    # KPIs Section
    html.Div(className='kpi-container', children=[
        html.Div(className='kpi', children=[
            html.Div(f"{total_customers}", className='kpi-number'),
            html.Div("Total Customers", className='kpi-label')
        ]),
        html.Div(className='kpi', children=[
            html.Div(f"{churned_customers}", className='kpi-number'),
            html.Div("Churned Customers", className='kpi-label')
        ]),
        html.Div(className='kpi', children=[
            html.Div(f"{churn_rate:.2f}%", className='kpi-number'),
            html.Div("Overall Churn Rate", className='kpi-label')
        ])
    ]),

    # Visualizations Section
    html.Div(className='row', children=[
        html.Div(className='column', children=[
            dcc.Graph(figure=fig_contract)
        ]),
        html.Div(className='column', children=[
            dcc.Graph(figure=fig_tenure)
        ])
    ])
])

# NEW CORRECT CODE
if __name__ == '__main__':
    app.run(debug=True)