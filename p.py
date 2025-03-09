# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Function to add custom CSS for background image
def add_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh; /* Full viewport height */
            color: white; /* Text color */
        }}
        .overlay {{
            background-color: rgba(0, 0, 0, 0.5); /* Black overlay with transparency */
            padding: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Specify the path to your image
image_path = "https://img.freepik.com/premium-photo/black-background-with-orange-blue-lines-orange-blue-stripes_994023-204360.jpg?semt=ais_hybrid"  # Change this to your image path
add_background_image(image_path)

# Load your data
data = pd.read_csv('Housing.csv')
data = data.dropna().drop_duplicates()

# Define the model and scaler
# model = LinearRegression()
# joblib.dump(model,'model.pkl')
# scaler = StandardScaler()
# joblib.dump(scaler,'scaler.pkl')
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
# Select numerical features for scaling
numerical_features = ['area', 'bedrooms', 'stories']
scaler.fit(data[numerical_features])
X = scaler.transform(data[numerical_features])
y = data['price']

# Train the model
model.fit(X, y)

# Sidebar description
st.sidebar.header("About the Project")
st.sidebar.write("""
This Housing Price Prediction App allows users to input details about a property, such as area, number of bedrooms, bathrooms, and stories. 
Using a linear regression model, it predicts the house price based on the input features. Visualizations of training data distributions are also provided.
""")

# Add an image to the sidebar
sidebar_image_url = "data:image/jpeg;base64,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"  # Replace with your image URL
st.sidebar.image(sidebar_image_url, use_column_width=True)  # Add the image

# Main app content
st.markdown('<div class="overlay">', unsafe_allow_html=True)
st.title("Housing Price Prediction App")
st.write("Enter the property details to predict the price.")

# Input fields for user data entry
area = st.number_input("Enter the area of the house (sq ft):", min_value=500, max_value=10000, value=7420, step=100)
bedrooms = st.number_input("Enter the number of bedrooms:", min_value=1, max_value=10, value=4, step=1)
bathrooms = st.number_input("Enter the number of bathrooms:", min_value=1, max_value=5, value=2, step=1)
stories = st.number_input("Enter the number of stories:", min_value=1, max_value=5, value=3, step=1)

st.sidebar.selectbox("Is it near a main road?", ['Yes', 'No'])
st.sidebar.selectbox("Does it have a basement?", ['Yes', 'No'])
st.sidebar.selectbox("Does it have a guest room?", ['Yes', 'No'])

st.sidebar.selectbox("Is there parking available?", ['Yes', 'No'])

# Create DataFrame for the input data
input_data = pd.DataFrame({'area': [area], 'bedrooms': [bedrooms], 'stories': [stories]})

input_data = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedrooms],
    'stories': [stories],
})

# # Convert 'Yes'/'No' to binary
# input_data['main_road'] = input_data['main_road'].map({'Yes': 1, 'No': 0})
# input_data['basement'] = input_data['basement'].map({'Yes': 1, 'No': 0})
# input_data['guest_room'] = input_data['guest_room'].map({'Yes': 1, 'No': 0})
# input_data['parking'] = input_data['parking'].map({'Yes': 1, 'No': 0})

# Scale input data
input_data[numerical_features] = scaler.transform(input_data[numerical_features])

# Predict price when the button is clicked
if st.button("Predict Price"):
    predicted_price = model.predict(input_data)
    st.success(f"Predicted Price: ${predicted_price[0]:,.2f}")
    
    # Display the histogram
    st.write("### Distribution of Training Data Features")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # Create a 2x2 grid for histograms
    axes = axes.ravel()  # Flatten the 2D array of axes to iterate over it
    
    # Plot histograms for each column in the data
    for i, col in enumerate(numerical_features + ['price']):
        data[col].hist(ax=axes[i], bins=15, edgecolor='yellow')
        axes[i].set_title(f'Distribution of {col}')
    
    plt.tight_layout()
    st.pyplot(fig)  # Pass the fig object to st.pyplot

    # Scatter plot of Predicted Price vs Area
    st.write("### Scatter Plot of Predicted Price vs Area")
    plt.figure(figsize=(8, 6))
    plt.scatter(data['area'], data['price'], color='blue', alpha=0.5, label='Training Data')
    plt.scatter(area, predicted_price[0], color='red', label='Predicted Price', s=100)
    plt.title('Predicted Price vs Area')
    plt.xlabel('Area (sq ft)')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid()
    st.pyplot(plt)  # Pass the entire plt object to st.pyplot
    st.markdown('</div>', unsafe_allow_html=True)
    
    