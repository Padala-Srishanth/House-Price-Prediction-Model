Housing Price Prediction & Analysis Platform
A comprehensive real estate intelligence platform built with Streamlit, Scikit-learn, and Python. This project goes beyond simple price prediction by offering advanced features including air quality monitoring, commute time calculation, mortgage planning, and AI-powered negotiation assistance.
Table of Contents

Overview
Features
Technologies Used
Machine Learning Model
Installation
Usage
Project Structure
API Integration
Screenshots
Future Enhancements
Contributing
License

Overview
This advanced housing price prediction platform combines machine learning with real-world real estate tools to provide users with comprehensive insights for property buying decisions. The application features an intuitive interface built with Streamlit and leverages multiple data sources to deliver accurate predictions and actionable intelligence.
The primary goal is to empower users with data-driven insights for making informed real estate decisions, from price prediction to financial planning and negotiation strategies.
Features
Core Features

House Price Prediction: Real-time price predictions based on property features
Data Visualizations:

Distribution analysis of training data features
Scatter plots comparing predicted vs actual prices
Interactive charts using Matplotlib and Plotly


Property Recommendation System: Cosine similarity-based recommendations for similar properties within budget
AI Real Estate Assistant: LLaMA-powered chatbot for real estate queries and property descriptions

Advanced Features

Air Quality Monitoring:

Real-time AQI data for 8+ major cities
PM2.5 and PM10 pollutant levels
Multiple area selections per city
Integration with Open-Meteo Air Quality API


Commute Time Calculator:

Travel time estimation to business districts
Distance and traffic condition analysis
Optimal travel time recommendations
Monthly commute cost estimates
Google Maps API integration support


Comprehensive Mortgage Calculator:

EMI Calculator: Monthly payment computation with interest breakdown
Loan Eligibility Checker: FOIR ratio analysis and maximum loan calculation
Amortization Schedule: Detailed year-by-year payment breakdown
Interactive Plotly visualizations
Downloadable payment schedules (CSV)


Price Negotiation Assistant:

Market condition-based negotiation strategies
Optimal offer range calculations
Property condition adjustments
Days-on-market analysis
Comparable property analysis
Proven negotiation tactics database



User Experience

Responsive Design: Custom CSS styling with background images
Interactive UI: Streamlit widgets for seamless interaction
Multi-tab Interface: Organized feature sections for easy navigation
Real-time Updates: Instant feedback with balloons, toasts, and spinners
Session Management: Maintains conversation history and prediction data

Technologies Used
Programming & Frameworks

Python 3.8+: Core programming language
Streamlit: Web application framework
Scikit-learn: Machine learning library

Data Processing & Analysis

Pandas: Data manipulation and analysis
NumPy: Numerical computing

Visualization

Matplotlib: Static plots (distributions, scatter plots)
Plotly: Interactive charts (pie charts, bar graphs, gauges)
Seaborn: Statistical data visualization

Machine Learning & Model Persistence

Scikit-learn StandardScaler: Feature normalization
Joblib: Model and scaler serialization (model.pkl, scaler.pkl)

API Integration

Requests: HTTP library for API calls
Open-Meteo Air Quality API: Real-time air quality data
Google Maps Distance Matrix API: Commute time calculations (optional)
Groq API (LLaMA 3.3): AI-powered chatbot responses

Additional Libraries

datetime: Time-based calculations and market timing analysis
cosine_similarity: Property recommendation system

Machine Learning Model
Algorithms Implemented
The model is trained and evaluated using multiple regression algorithms:

Linear Regression:

Provides baseline linear relationships between features and prices
Fast prediction and easy interpretability
Best for linearly separable data


Decision Tree Regressor:

Handles complex non-linear relationships
Captures feature interactions naturally
No assumption about data distribution


Random Forest Regressor:

Ensemble method combining multiple decision trees
Reduces overfitting through bagging
Higher accuracy through variance reduction
Selected as final model due to superior performance



Model Features

Input Features:

area (square feet)
bedrooms (number)
stories (number)
Additional categorical features: city, furnishing, amenities


Target Variable: price (USD)
Preprocessing:

StandardScaler normalization for numerical features
Missing value handling
Duplicate removal
Outlier detection and treatment


Model Persistence:

Trained model saved as model.pkl
Feature scaler saved as scaler.pkl
Easy loading for real-time predictions



Model Evaluation

Cross-validation techniques applied
Performance metrics: RMSE, MAE, R² score
Feature importance analysis
Prediction vs actual price visualization
