#Import Necessary Libraries
import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import pickle
from streamlit_option_menu import option_menu
from streamlit_navigation_bar import st_navbar
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn import preprocessing 

# Create sidebar menu
with st.sidebar:
    st.write(f'<h1 style="text-align: center;">Customer Mall Spending Score</h1>', unsafe_allow_html=True)

    selected_tab = option_menu(
            menu_title=None,  # required
            # options=["Problem statement", "Exploratory data analysis", "Model Training", "Prediction","Prescriptive Analysis"],
            options=["Problem statement", "Exploratory data analysis", "Model Training", "Prediction",""],
            icons=["house", "bar-chart", "search", "envelope"], 
            orientation="vertical",
            default_index=0
        )
#Import dataset
df = pd.read_csv('data\Mall_Customers.csv')
# st.info('1. Dataset Imported', icon='\u2705')
df = pd.read_csv('data\Mall_Customers.csv')

#Perform Linear Regression
def linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    error = mean_squared_error(y_test, prediction)
    return error,prediction

#Perform Polynomial Regression
def polynomial_regression(X_train, y_train, X_test, y_test, degree=2):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    prediction = model.predict(X_poly_test)
    error = mean_squared_error(y_test, prediction)
    return error,prediction

#Perform Ridge Regression
def ridge_regression(X_train, y_train, X_test, y_test):
    model = Ridge()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    error = mean_squared_error(y_test, prediction)
    return error,prediction

#Perform Lasso Regression
def lasso_regression(X_train, y_train, X_test, y_test):
    model = Lasso()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    error = mean_squared_error(y_test, prediction)
    return error,prediction

#Perform Elastic Net
def elastic_net_regression(X_train, y_train, X_test, y_test):
    model = ElasticNet()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    error = mean_squared_error(y_test, prediction)
    return error,prediction
#Perform SVM Regression
def svm_regression(X_train, y_train, X_test, y_test):
    model = SVR()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    error = mean_squared_error(y_test, prediction)
    return error,prediction    
    
if selected_tab == "Problem statement":
    #Import File
    st.markdown(f'<h2>Problem statement</h2>', unsafe_allow_html=True)
    st.write('Identify individuals who are likely to spend more money when visiting a mall.')

    st.write(f'<h2>Solution</h2>', unsafe_allow_html=True)
    st.write('Machine learning algorithms can combine income and spend score data to create precise customer segments, enabling highly targeted and personalized marketing campaigns. This ensures dynamic and effective engagement with various customer groups.')

    st.markdown(f'<h2>Benefits</h2>', unsafe_allow_html=True)
    st.write('Revenue Growth: By leveraging income and spend score data, businesses can optimize their marketing efforts to drive higher revenue growth.')
    st.write('Cost Efficiency: Targeted marketing reduces wasted spend on ineffective campaigns, leading to more efficient use of marketing budgets.')
    st.write('Competitive Advantage: Businesses that effectively use this data gain a competitive edge by being more responsive to customer needs and market dynamics.')

if selected_tab == "Exploratory data analysis": 
    st.markdown(f'<h1 style="text-align: center;">Exploratory data analysis</h1>', unsafe_allow_html=True)

    st.write('Few Records from file')
    st.table(df.head())

    st.info('3. EDA', icon='\u2705')
    st.write('Few Statistical details from file')
    st.table(df.describe())

    st.write('Dimension of file')
    df.shape

    st.write('Data Types')
    st.table(df.dtypes)

    st.write('## Data Cleaning & Preprocessing')
    # Encode gender column
    st.info('2. Gender encoding complete', icon='\u2705')
    label_encoder = preprocessing.LabelEncoder()  
    df['Gender']= label_encoder.fit_transform(df['Gender']) 

    # Rename Columns
    st.info('3. Columns renamed', icon='\u2705')
    st.write('Annual Income (k$) changed to Annual Income')
    st.write('Spend1ing Score (1-100) changed to  Spending Score')
    df.rename(columns={'Annual Income (k$)':'Annual Income'},inplace=True)
    df.rename(columns={'Spending Score (1-100)':'Spending Score'},inplace=True)
    st.table(df.head())
    st.write('## Visualizations:')
    st.write('Barplot to identify outliers')
    
    #1. Barplot
    st.write('1. Barplot')
    col1,col2 = st.columns(2)
    with col1:
        fig = plt.figure(figsize=(8,8))
        sb.barplot(df)
        st.pyplot(fig)
    with col2:
        st.write('No outliers found')   

    #2. Histograms
    st.write('## 2. Histograms for Age, Annual Income, and Spending Score')
    col1,col2 = st.columns(2)
    with col1:
        st.write('# EDA Summary')
        st.write('i. Most customers are between 25 and 50 years old.')
        st.write('ii. Annual Income Distribution: The income ranges from 15k to 137k, with many customers having an income between 40k and 80k.')
        st.write('iii. Spending Score Distribution: Scores are well-distributed across the range, indicating a mix of low to high spenders.')
    with col2:
        col1,col2,col3 = st.columns(3)
        with col1:
            fig = plt.figure(figsize=(4,4))
            sb.histplot(df['Age'],bins=50)
            plt.xlabel='Age'
            plt.ylabel='Values'
            plt.title='Age Distribution'
            st.pyplot(fig)
        with col2:
            fig = plt.figure(figsize=(4,4))
            sb.histplot(df['Spending Score'],bins=50)
            plt.xlabel='Spending Score'
            plt.ylabel='Values'
            plt.title='Spending Distribution'
            st.pyplot(fig)
        with col3:
            fig = plt.figure(figsize=(4,4))
            sb.histplot(df['Annual Income'],bins=50)
            plt.xlabel='Annual Income'
            plt.ylabel='Values'
            plt.title='Income Distribution'
            st.pyplot(fig)
        
    #3. Scatterplot
    st.write('## 3. Scatter plots for relationships')
    col1,col2 = st.columns(2)
    with col1:
        st.write('i. Age vs. Annual Income: No strong relationship, customers of various ages have a wide range of incomes.')
        st.write('ii. Age vs. Spending Score: No clear trend, indicating spending score is not directly related to age.')
        st.write('iii. Annual Income vs. Spending Score: Some clusters suggest that spending score may be influenced by income, but the relationship is not linear.')
    with col2:
        col1,col2,col3 = st.columns(3)
        with col1:
            fig = plt.figure(figsize=(4,4))
            sb.scatterplot(x=df['Age'],y=df['Annual Income'])
            st.pyplot(fig)
        with col2:
            fig = plt.figure(figsize=(4,4))
            sb.scatterplot(x=df['Age'],y=df['Spending Score'])
            st.pyplot(fig)
        with col3:
            fig = plt.figure(figsize=(4,4))
            sb.scatterplot(x=df['Annual Income'],y=df['Spending Score'])
            st.pyplot(fig)
    
    # 4. Correalation Heatmap
    st.write('4. Correalation Heatmap')
    st.write('Age, Annual Income, and Spending Score: Weak correlations between these variables, suggesting that each feature adds unique information about the customers.')
    fig, ax = plt.subplots(figsize=(15, 8))
    sel_cols = df[['Gender','Age','Annual Income','Spending Score']].corr()
    plt.figure(figsize=(15,8))
    sb.heatmap(sel_cols, annot=True, ax=ax)
    st.pyplot(fig)

    #5. Line Graph
    st.write('5. Line Graph')
    fig = plt.figure(figsize=(4,4))
    plt.plot(df['Annual Income'],df['Spending Score'])
    plt.figure(figsize=(15,8))
    st.pyplot(fig)

    # 6. Count plot for gender
    st.write('6. Count plot for gender')
    st.write('The dataset has a relatively balanced gender distribution.')
    fig = plt.figure(figsize=(4,4))
    sb.countplot(x=df['Gender'],data=df)
    plt.figure(figsize=(15,8))
    st.pyplot(fig)


if selected_tab == 'Model Training':
    st.write('## Data Cleaning & Preprocessing')
    # Encode gender column
    st.info('2. Gender encoding complete', icon='\u2705')
    label_encoder = preprocessing.LabelEncoder()  
    df['Gender']= label_encoder.fit_transform(df['Gender']) 

    # Rename Columns
    st.info('3. Columns renamed', icon='\u2705')
    st.write('Annual Income (k$) changed to Annual Income')
    st.write('Spend1ing Score (1-100) changed to  Spending Score')
    df.rename(columns={'Annual Income (k$)':'Annual Income'},inplace=True)
    df.rename(columns={'Spending Score (1-100)':'Spending Score'},inplace=True)
    st.table(df.head())
    
    #Train Split Test
    st.info('4. Train Split Test', icon='\u2705')
    x=df[['Age']]
    y=df['Spending Score']
    st.write("Shape of x:", x.shape)
    st.write("Shape of y:", y.shape)

    st.info('5. Fit Models', icon='\u2705')
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)
    st.write('## Executing Linear Regression', icon='\u2705')
    error,_  = linear_regression(x_train,y_train,x_test,y_test)
    with open('predictions.pkl', 'wb') as f:
        pickle.dump(_, f)
    st.write('Root Mean Squared Error', round(np.sqrt(error),2))
    st.write('## Executing polynomial_regression', icon='\u2705')
    error,_  = polynomial_regression(x_train,y_train,x_test,y_test)
    st.write('Root Mean Squared Error', round(np.sqrt(error),2))
    st.write('## Executing Ridge Regression', icon='\u2705')
    error,_  = ridge_regression(x_train,y_train,x_test,y_test)
    st.write('Root Mean Squared Error', round(np.sqrt(error),2))
    st.write('## Executing Lasso Regression', icon='\u2705')
    error,_  = lasso_regression(x_train,y_train,x_test,y_test)
    st.write('Root Mean Squared Error', round(np.sqrt(error),2))
    st.write('## Executing Elastic Net Regression', icon='\u2705')
    error,_  = elastic_net_regression(x_train,y_train,x_test,y_test)
    st.write('Root Mean Squared Error', round(np.sqrt(error),2))
    st.write('## Executing Support Vector Machine', icon='\u2705')
    error,_ = svm_regression(x_train,y_train,x_test,y_test)
    st.write('Root Mean Squared Error', round(np.sqrt(error),2))
    st.write('Considering linear regression model after analysing the error which is saved on file predictions.pkl', icon='\u2705')
if selected_tab == 'Prediction':
    df.rename(columns={'Annual Income (k$)':'Annual Income'},inplace=True)
    df.rename(columns={'Spending Score (1-100)':'Spending Score'},inplace=True)
    # Extract features and target variable
x = df[['Age']]
y = df['Spending Score']

# User input
age = st.number_input("Input age for prediction:")

# Button for prediction
if st.button("Predict"):
    # Train the model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Prepare the input for prediction
    xtest1 = np.array([[age]])  # Reshape to 2D array for a single prediction

    # Make prediction
    new_predict = model.predict(xtest1)

    # Display the result
    st.write(f"Predicted Spending Score: {new_predict[0]:.2f}")