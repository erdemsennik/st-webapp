import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder

# # Title
# st.title('Webapp using Streamlit')

# st.header('Case study on Telco dataset')
# # Image
# st.image("Normalized Stacked Pie-Chart&Churn.jpg", width = 500)

data = pd.read_csv('df_end.csv')

menu = st.sidebar.radio('Menu', ['Introduction', 'Graphs', 'Churn Prediction'])
if menu =='Introduction':
    # Title
    st.title('A Web-App Study for Predicting Telco Customer Churn')
    st.write("Forecasting customer churn is pivotal for telecommunication companies in effectively retaining their customer base. The expense of acquiring new customers surpasses that of retaining existing ones. Consequently, major telecommunications corporations are actively pursuing the development of predictive models to identify customers who are more likely to switch providers, enabling proactive measures to be taken in response.")
    st.write("Within this web application, an analysis of some features from the dataset was conducted to assess their influence on the likelihood of customer churn.")
    ## Image
    st.image("Normalized Stacked Pie-Chart&Churn.jpg", width = 500)
    ## Explanation
    st.write("The pie-chart above shows the distribution of observations across the response variable's classes: **No** and **Yes**. This dataset exhibits an imbalance since both classes are not evenly distributed among all observations. The no class constitutes the majority (73.5%). This imbalance can result in a significant number of false negatives during modeling.")
    
    
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
    


    st.header("Tabular dataof a df_end")
    if st.checkbox("Tabular Data"):
        st.table(data.head(15))
    st.header('Statistical Summary of dataframe')
    if st.checkbox('Statistic'):
        st.table(data.describe())
    
    st.title('Graphs')
    

if menu =='Churn Prediction':
    st.title('Churn Prediction')

     
    le_con = LabelEncoder()
    le_dep = LabelEncoder()
    le_mar = LabelEncoder()
    le_pre = LabelEncoder()
    le_ref = LabelEncoder()
    data['Contract'] = le_con.fit_transform(data['Contract'])
    data['Dependents'] = le_dep.fit_transform(data['Dependents'])
    data['Married'] = le_mar.fit_transform(data['Married'])
    data['Premium Tech Support'] = le_pre.fit_transform(data['Premium Tech Support'])
    data['Referred a Friend'] = le_ref.fit_transform(data['Referred a Friend'])
   
    
    ## Create a rf
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    # select independent variables
    X = data.drop(columns='Churn')
    # select dependent variables
    y = data.loc[:, 'Churn']

    

    mdl = rf.fit(X, y)

    Age = np.arange(81, 19, -1).tolist()
    age = st.selectbox("Age", Age)

    Number_of_Referrals = np.arange(0, 12, 1).tolist()
    number_of_referrals = st.selectbox("Number of Referrals", Number_of_Referrals)

    Contract = ('Month-to-month', 'One year', 'Two year')
    contract = st.selectbox("Contract", Contract)

    Dependents = ('Yes', 'No')
    dependents = st.selectbox("Dependents", Dependents)

    Married = ('Yes', 'No')
    married = st.selectbox("Married", Married)

    Premium_Tech_Support = ('Yes', 'No')
    premium_tech_support = st.selectbox("Premium_Tech_Support", Premium_Tech_Support)

    Referred_a_Friend = ('Yes', 'No')
    referred_a_friend = st.selectbox("Reffered a Friend", Referred_a_Friend)
    

    ok = st.button("Predict Churn Possibility")
    if ok:

        

        Xtest = np.array([[age, number_of_referrals, contract, dependents, married, premium_tech_support, referred_a_friend]])
        Xtest[:, 2] = le_con.transform(Xtest[:, 2])
        Xtest[:, 3] = le_dep.transform(Xtest[:, 3])
        Xtest[:, 4] = le_mar.transform(Xtest[:, 4])
        # X[:, 3] = le_offer.transform(Xtest[:, 3])
        Xtest[:, 5] = le_pre.transform(Xtest[:, 5])
        Xtest[:, 6] = le_ref.transform(Xtest[:, 6])
        # X[:, 6] = le_udata.transform(Xtest[:, 6])

        # X = X.astype(float)
        # st.write(f'{Xtest[:, 2]}')
        # st.write(f'{Xtest}')
        
   
        # prediction = data['model'].predict(X)
        prediction = mdl.predict(Xtest)
        

        from sklearn.metrics import accuracy_score
        # accuracy = accuracy_score(y, mdl.predict(Xtest))

        # st.write(f'{prediction}')
        if prediction == ['No']:
            st.write(f"The customer will more likely not Churn.")
        else:
            st.write(f"The customer will more likely Churn.")

 
        


  
