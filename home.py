import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu

# streamlit run Streamlit.py

st.set_page_config(
    page_title="performance",
    page_icon="👨‍💻",
)

selected = option_menu(
    menu_title=None,
    options=["Home", "Project","Contact us"],
    default_index=0,
    orientation="horizontal"
)

if selected == "Home":
    st.title("Home Page")

    st.write(
         '''In today's competitive landscape, organizations are constantly seeking ways to optimize their workforce and improve overall performance. One key aspect of this endeavor is understanding what factors contribute to employee performance and how predictive models can aid in talent management. By leveraging advanced analytics techniques and utilizing vast amounts of employee data, organizations can develop models that accurately predict employee performance and identify the underlying factors that drive success within their workforce.'''
         '''This project aims to build such a predictive model to assess employee performance and uncover the critical elements that influence it. By analyzing various data points such as performance reviews, training records, and demographic information, we can gain valuable insights into what separates high-performing employees from their counterparts. Through this analysis, organizations can make informed decisions regarding talent acquisition, development, and retention strategies.'''
               )
    st.write(
         '''This project aims to build such a predictive model to assess employee performance and uncover the critical elements that influence it. By analyzing various data points such as performance reviews, training records, and demographic information, we can gain valuable insights into what separates high-performing employees from their counterparts. Through this analysis, organizations can make informed decisions regarding talent acquisition, development, and retention strategies.'''

               )
    st.write(
         ''' predictive model developed in this project will serve as a powerful tool for talent management, enabling organizations to identify and nurture high-potential employees while addressing any areas of concern that may hinder performance. By understanding the characteristics and behaviors associated with top performers, organizations can tailor their recruitment efforts, training programs, and performance management initiatives to optimize workforce effectiveness and drive organizational success.'''
               )
    st.write(''' this project represents a proactive approach to talent management, leveraging data-driven insights to cultivate a high-performing workforce. '''
             )

    def plot_heatmap(correlation_matrix):
        """
        Function to plot a heatmap for the correlation matrix.
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        return plt
    
    # Load employee data
    employee_data=pd.read_csv("employee_data.csv")

    # Drop irrelevant columns
    employee_data = employee_data.drop(['Performance', 'EmployeeNumber','DailyRate','HourlyRate','MonthlyRate','PerformanceRating','WorkLifeBalance'], axis=1)

    # Calculate the correlation matrix
    correlation_matrix = employee_data.corr()

    # Plot the correlation matrix as a heatmap
    st.subheader('Correlation Matrix Heatmap')
    st.pyplot(plot_heatmap(correlation_matrix))
    
    # Calculate the correlation matrix
    correlation_matrix = employee_data.corr()

    # Display the correlation matrix
    st.subheader("Correlation Matrix")
    st.write(correlation_matrix)


    

if selected == "Project":
    st.title("Project Page")
    
    # Load employee data
    @st.cache_data
    def load_data():
        """
        Function to load employee data from a CSV file.
        """
        return pd.read_csv('employee_data.csv')

    # Data preprocessing (replace this with your actual preprocessing steps)
    # For this example, let's assume preprocessing involves handling missing values
    def preprocess_data(data):
        """
        Placeholder function for data preprocessing.
        """
        return data.fillna(data.mean())

    # Define features and target variable
    def define_features_target(data):
        """
        Function to define features and target variable.
        """
        X = data.drop(['Performance', 'EmployeeNumber','DailyRate','HourlyRate','MonthlyRate','PerformanceRating','WorkLifeBalance'], axis=1)  # Assuming 'Performance' is the target variable
        y = data['Performance']
        return X, y

    # Sidebar for input features
    def sidebar_input_features(X):
        """
        Function to create sidebar inputs for each feature.
        """
        input_features = {}
        for feature in X.columns:
            input_features[feature] = st.number_input(f'Enter {feature}',value=None)
        return input_features

    # Predict performance based on input features
    def predict_performance(model, input_data):
        """
        Function to predict performance based on input features.
        """
        return model.predict(input_data)

    # Convert sidebar inputs into a DataFrame
    def convert_sidebar_inputs(input_features):
        """
        Function to convert sidebar inputs into a DataFrame.
        """
        return pd.DataFrame([input_features])

    # Display raw data (optional)
    def display_raw_data(data):
        """
        Function to display raw data.
        """
        st.subheader('Raw Data')
        st.write(data)

    st.title('Employee Performance Prediction')

    # Load data
    employee_data = load_data()

    # Data preprocessing
    employee_data = preprocess_data(employee_data)

    # Define features and target variable
    X, y = define_features_target(employee_data)

    # Display raw data
    if st.checkbox('Show Raw Data'):
        display_raw_data(employee_data)

    # Input features
    st.header('Input Features')
    input_features = sidebar_input_features(X)

    # Create a train-test split (if needed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model (you can replace RandomForestClassifier with any other model)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make prediction based on input features
    if st.button('Predict Performance'):
        input_data = convert_sidebar_inputs(input_features)
        prediction = predict_performance(model, input_data)
        st.write(f'Predicted Performance: {prediction[0]}')
    
    
    


if selected == "Contact us":
    contact_from="""
    <form action="https://formsubmit.co/krrishshekhaliya7@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="your name" required>
        <input type="email" name="email" placeholder="your email" required>
        <textarea name="message" placeholder="Details of your problem"></textarea>
        <button type="submit">Send</button>
    </form>
    """

    st.markdown(contact_from,unsafe_allow_html=True)

    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
    local_css("style\style.css")

