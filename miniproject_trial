import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import LabelEncoder 
# import matplotlib.pyplot as plt 
# import seaborn as sns
# import math 

# # Title
# st.title('Webapp using Streamlit')

# st.header('Case study on Telco dataset')
# # Image
# st.image("Normalized Stacked Pie-Chart&Churn.jpg", width = 500)

data = pd.read_csv('df_end.csv')
# raw_data = pd.read_csv('df_last.csv')
# st.write("Shape of dataset", data.shape)
# st.write(data.head(5))
menu = st.sidebar.radio('Menu', ['Introduction', 'Graphs', 'Churn Prediction'])
if menu =='Introduction':
    # Title
    st.title('A Web-App Study for Predicting Telco Customer Churn')
    st.write("Forecasting customer churn is pivotal for telecommunication companies in effectively retaining their customer base. The expense of acquiring new customers surpasses that of retaining existing ones. Consequently, major telecommunications corporations are actively pursuing the development of predictive models to identify customers who are more likely to switch providers, enabling proactive measures to be taken in response.")
    st.write("Within this web application, an analysis of some features from the dataset was conducted to assess their influence on the likelihood of customer churn.")
    # Image
    st.image("Normalized Stacked Pie-Chart&Churn.jpg", width = 500)
    # Explanation
    st.write("The pie-chart above shows the distribution of observations across the response variable's classes: **No** and **Yes**. This dataset exhibits an imbalance since both classes are not evenly distributed among all observations. The no class constitutes the majority (73.5%). This imbalance can result in a significant number of false negatives during modeling.")
    # data = pd.read_csv('df_end.csv')
    # st.write("Shape of dataset", data.shape)
    # st.write(data.head(5))
    # st.subheader('Describe What your aim is')
    
if menu =='Graphs':
    st.header('Stacked Graphs')
    st.write('A normalized stacked bar plot ensures uniform column height. Nevertheless, it excels in illustrating the variation of the response variable across different groups of an independent variable.')
    st.markdown("**o** The churn rate of the customer is increasing by increment of age when looking at the graph below. That might show that it is difficult to pertain elder customers into the system for a long time.")
    st.image('Stacked_Age&Churn.jpg',width=600)
    st.markdown("**o** The customer churn rate significantly decreases with an increasing number of referrals, as evident from the graph below. To maintain a customer retention rate of approximately 90%, it appears that a minimum of four referrals will be necessary. A campaign encouraging customers to refer a suitable friend can not only bring a new person into the system but also help retain existing customers for longer periods.")
    st.image('Stacked_Number of Referrals&Churn.jpg',width=600)
    st.header('Stacked Graph for Categorical data')
    st.markdown('**o** Customers who have contracts exceeding one year exhibit lower churn rates. The churn rate significantly diminishes when customers opt for a two-year contract, nearly reaching negligible levels. Providing discounted annual campaigns to customers can contribute to retaining them for extended durations.')
    st.image('Stacked_Contract&Churn.jpg',width=600)
    st.markdown('**o** Married customers tend to display lower churn rates. This trend might also be directly correlated with the presence of dependents, which indicates whether the customer resides with any dependents, such as children, parents, grandparents, and so forth.')
    st.image('Stacked_Dependents&Churn.jpg',width=600)
    st.image('Stacked_Married&Churn.jpg',width=600)
    st.markdown('**o** The availability of technical support appears to correlate with a reduction in the churn rate among customers. The churn rate demonstrates a notable decrease of almost 15% when customers have access to technical support. Surveys related to technical support requests can be conducted with customers.S ')
    st.image('Stacked_Premium Tech Support&Churn.jpg',width=600)
    st.image('Stacked_Referred a Friend&Churn.jpg',width=600)
    st.markdown('**o** Customers who have contracts')
    

# st.image('Y.jpg',width=550)
    st.header("Tabular dataof a df_end")
    if st.checkbox("Tabular Data"):
        st.table(data.head(15))
    st.header('Statistical Summary of dataframe')
    if st.checkbox('Statistic'):
        st.table(data.describe())
    # if st.header("Correlation Graph"):
    #     fig, ax = plt.subplots(figsize=(5,2.5))
    #     # sns.heatmap(data.corr(),annot=Tru, cmap='coolwarm')
    #     st.pyplot(fig)
    st.title('Graphs')
    # graph = st.selectbox('Different type of graphs',['Scatter', 'Bar', 'Histogram'])
    # if graph =='Scatter':
    #     value = st.slider('Filter data using carat', 0, 6)
    #     data = data.loc[data['carat']>=value]
    #     fig , ax = plt.subplots(figsize=(10,5))
    #     sns.scatterplot(data = data, x='carat', y='price', hue='cut')
    #     st.pyplot(fig)
    # if graph =='Bar':
    #     fig, ax = plt.subplots(figsize=(3.5,2))
    #     sns.barplotplot(x='cut', y=data.cut.index, data = data)
    #     st.pyplot(fig)
    # if graph =='Histogram':
    #     fig, ax = plt.subplots(figsize=(5,3))
    #     sns.displot(data.price, kde=True)
    #     st.pyplot(fig)

