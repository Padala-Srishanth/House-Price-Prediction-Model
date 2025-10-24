# -------------------------------------------
# Import necessary libraries
# -------------------------------------------
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import joblib
import requests
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px


# -------------------------------------------
# AQI Helper Functions
# -------------------------------------------
def get_air_quality(lat, lon):
    """
    Fetch current AQI and pollutants for given coordinates using Open-Meteo API.
    """
    try:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "us_aqi,pm2_5,pm10"
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            return {"error": f"API returned status code {response.status_code}"}
        
        data = response.json()
        
        # Try current data first (more reliable)
        if "current" in data:
            aqi = data["current"].get("us_aqi")
            pm25 = data["current"].get("pm2_5")
            pm10 = data["current"].get("pm10")
            
            # Check if all values are valid
            if aqi is not None and pm25 is not None and pm10 is not None:
                return {
                    "aqi": round(aqi),
                    "pm2_5": round(pm25, 2),
                    "pm10": round(pm10, 2)
                }
        
        # Fallback to hourly data
        if "hourly" in data and "us_aqi" in data["hourly"]:
            aqi_list = data["hourly"]["us_aqi"]
            pm25_list = data["hourly"]["pm2_5"]
            pm10_list = data["hourly"]["pm10"]
            
            # Find the last non-None value
            for i in range(len(aqi_list) - 1, -1, -1):
                if aqi_list[i] is not None and pm25_list[i] is not None and pm10_list[i] is not None:
                    return {
                        "aqi": round(aqi_list[i]),
                        "pm2_5": round(pm25_list[i], 2),
                        "pm10": round(pm10_list[i], 2)
                    }
        
        return {"error": "No valid air quality data available for this location"}
        
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def aqi_level(aqi):
    """Return AQI level with color indicator"""
    if aqi <= 50:
        return "🟢 Good"
    elif aqi <= 100:
        return "🟡 Moderate"
    elif aqi <= 150:
        return "🟠 Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "🔴 Unhealthy"
    elif aqi <= 300:
        return "🟣 Very Unhealthy"
    else:
        return "⚫ Hazardous"


# -------------------------------------------
# Commute Time Calculator Functions
# -------------------------------------------
def calculate_commute_time(origin, destination, api_key=None):
    """
    Calculate travel time and distance using Google Maps Distance Matrix API
    """
    if not api_key or api_key == "YOUR_GOOGLE_MAPS_API_KEY":
        # Demo data when API key is not provided
        demo_routes = {
            "Hyderabad-Hitech City": {"duration": 25, "distance": 15.2, "traffic": "Moderate"},
            "Delhi-Connaught Place": {"duration": 35, "distance": 18.5, "traffic": "Heavy"},
            "Mumbai-BKC": {"duration": 40, "distance": 22.1, "traffic": "Heavy"},
            "Bangalore-Whitefield": {"duration": 45, "distance": 25.3, "traffic": "Heavy"},
        }
        
        for key in demo_routes:
            if origin.lower() in key.lower() or destination.lower() in key.lower():
                return demo_routes[key]
        
        return {"duration": 30, "distance": 12.5, "traffic": "Moderate"}
    
    try:
        url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        params = {
            "origins": origin,
            "destinations": destination,
            "key": api_key,
            "departure_time": "now",
            "traffic_model": "best_guess"
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data["status"] == "OK":
            element = data["rows"][0]["elements"][0]
            if element["status"] == "OK":
                duration = element["duration_in_traffic"]["value"] / 60  # Convert to minutes
                distance = element["distance"]["value"] / 1000  # Convert to km
                
                # Estimate traffic based on duration
                traffic = "Light" if duration < 20 else "Moderate" if duration < 40 else "Heavy"
                
                return {
                    "duration": round(duration),
                    "distance": round(distance, 1),
                    "traffic": traffic
                }
        
        return None
    except Exception as e:
        return None


# -------------------------------------------
# Mortgage Calculator Functions
# -------------------------------------------
def calculate_emi(principal, annual_rate, tenure_years):
    """Calculate EMI using standard formula"""
    monthly_rate = annual_rate / (12 * 100)
    n_months = tenure_years * 12
    
    if monthly_rate == 0:
        return principal / n_months
    
    emi = principal * monthly_rate * ((1 + monthly_rate) ** n_months) / (((1 + monthly_rate) ** n_months) - 1)
    return round(emi, 2)


def check_loan_eligibility(monthly_income, existing_emi, requested_loan, annual_rate, tenure_years):
    """Check loan eligibility based on income"""
    new_emi = calculate_emi(requested_loan, annual_rate, tenure_years)
    total_emi = existing_emi + new_emi
    
    # FOIR (Fixed Obligation to Income Ratio) should be less than 50%
    foir = (total_emi / monthly_income) * 100
    
    max_eligible_emi = monthly_income * 0.5 - existing_emi
    max_eligible_loan = max_eligible_emi * (((1 + annual_rate/(12*100)) ** (tenure_years*12)) - 1) / (annual_rate/(12*100) * ((1 + annual_rate/(12*100)) ** (tenure_years*12)))
    
    return {
        "eligible": foir <= 50,
        "foir": round(foir, 2),
        "new_emi": round(new_emi, 2),
        "total_emi": round(total_emi, 2),
        "max_eligible_loan": round(max_eligible_loan, 2)
    }


def generate_amortization_schedule(principal, annual_rate, tenure_years):
    """Generate amortization schedule"""
    monthly_rate = annual_rate / (12 * 100)
    n_months = tenure_years * 12
    emi = calculate_emi(principal, annual_rate, tenure_years)
    
    balance = principal
    schedule = []
    
    for month in range(1, n_months + 1):
        interest = balance * monthly_rate
        principal_paid = emi - interest
        balance -= principal_paid
        
        if balance < 0:
            balance = 0
        
        schedule.append({
            "Month": month,
            "EMI": round(emi, 2),
            "Principal": round(principal_paid, 2),
            "Interest": round(interest, 2),
            "Balance": round(balance, 2)
        })
    
    return pd.DataFrame(schedule)


# -------------------------------------------
# Price Negotiation Functions
# -------------------------------------------
def calculate_negotiation_range(predicted_price, market_conditions="Normal"):
    """Calculate optimal negotiation range"""
    
    # Market condition multipliers
    multipliers = {
        "Buyer's Market": 0.15,  # Can negotiate 15% below
        "Normal": 0.08,           # Can negotiate 8% below
        "Seller's Market": 0.03   # Can negotiate 3% below
    }
    
    negotiation_percentage = multipliers.get(market_conditions, 0.08)
    
    optimal_offer = predicted_price * (1 - negotiation_percentage)
    max_offer = predicted_price * (1 - negotiation_percentage/2)
    min_offer = predicted_price * (1 - negotiation_percentage * 1.5)
    
    return {
        "predicted_price": round(predicted_price, 2),
        "optimal_offer": round(optimal_offer, 2),
        "max_offer": round(max_offer, 2),
        "min_offer": round(min_offer, 2),
        "negotiation_room": round(predicted_price - optimal_offer, 2),
        "percentage_below": round(negotiation_percentage * 100, 2)
    }


def analyze_market_timing():
    """Analyze current market timing"""
    current_month = datetime.now().month
    
    # Best months to buy (typically less competition)
    best_months = [12, 1, 2]  # December, January, February
    good_months = [6, 7, 8]   # Mid-year
    
    if current_month in best_months:
        timing = "Excellent"
        advice = "Winter months typically see less buyer competition. Great time to negotiate!"
    elif current_month in good_months:
        timing = "Good"
        advice = "Mid-year can offer good deals. Sellers may be motivated before year-end."
    else:
        timing = "Fair"
        advice = "Spring/Fall tend to be competitive. Consider waiting or be prepared to act quickly."
    
    return {
        "timing": timing,
        "advice": advice,
        "current_month": datetime.now().strftime("%B %Y")
    }


# -------------------------------------------
# Function to add custom CSS for background
# -------------------------------------------
def add_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh;
            color: white;
        }}
        .overlay {{
            background-color: rgba(0, 0, 0, 0.55);
            padding: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------
# Load model and data
# -------------------------------------------
image_path = "https://img.freepik.com/premium-photo/black-background-with-orange-blue-lines-orange-blue-stripes_994023-204360.jpg"
add_background_image(image_path)

data = pd.read_csv('Housing.csv').dropna().drop_duplicates()
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

numerical_features = ['area', 'bedrooms', 'stories']
scaler.fit(data[numerical_features])
X = scaler.transform(data[numerical_features])
y = data['price']
model.fit(X, y)

# Prepare location encoding
location_map = {loc: idx for idx, loc in enumerate(data['city'].unique())} if 'city' in data.columns else {}
if 'city' in data.columns:
    data['city_code'] = data['city'].map(location_map)
else:
    data['city_code'] = 0


# -------------------------------------------
# Sidebar: Filters and Info
# -------------------------------------------
st.sidebar.header("Housing Price Filters")
city = st.sidebar.selectbox("Select City", ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"])
area_type = st.sidebar.selectbox("Area Type", ["Urban", "Semi-urban", "Rural"])
furnishing = st.sidebar.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
main_road = st.sidebar.selectbox("Near Main Road?", ['Yes', 'No'])
basement = st.sidebar.selectbox("Has Basement?", ['Yes', 'No'])
guest_room = st.sidebar.selectbox("Has Guest Room?", ['Yes', 'No'])
parking = st.sidebar.selectbox("Has Parking?", ['Yes', 'No'])

st.sidebar.markdown("---")

# -------------------------------------------
# Air Quality Checker Section in Sidebar
# -------------------------------------------
st.sidebar.subheader("🌫️ Air Quality Checker")

# City and area coordinates mapping
city_area_coords = {
    "Hyderabad": {
        "City Center": (17.3850, 78.4867),
        "Hitech City": (17.4485, 78.3908),
        "Gachibowli": (17.4399, 78.3489),
        "Secunderabad": (17.4399, 78.4983),
        "Madhapur": (17.4490, 78.3910),
        "Kukatpally": (17.4849, 78.3914),
        "Bachupally": (17.5264, 78.3699)
    },
    "Delhi": {
        "Connaught Place": (28.6289, 77.2065),
        "Dwarka": (28.5921, 77.0460),
        "Rohini": (28.7491, 77.0674),
        "Saket": (28.5244, 77.2066),
        "Noida": (28.5355, 77.3910),
        "Gurgaon": (28.4595, 77.0266)
    },
    "Mumbai": {
        "Andheri": (19.1136, 72.8697),
        "Bandra": (19.0596, 72.8295),
        "Dadar": (19.0176, 72.8481),
        "Powai": (19.1176, 72.9060),
        "Thane": (19.2183, 72.9781),
        "Navi Mumbai": (19.0330, 73.0297)
    },
    "Bangalore": {
        "Koramangala": (12.9352, 77.6245),
        "Whitefield": (12.9698, 77.7500),
        "Indiranagar": (12.9716, 77.6412),
        "Marathahalli": (12.9591, 77.7010),
        "Electronic City": (12.8456, 77.6603),
        "JP Nagar": (12.9082, 77.5855)
    },
    "Chennai": {
        "T Nagar": (13.0418, 80.2341),
        "Anna Nagar": (13.0878, 80.2085),
        "Velachery": (12.9750, 80.2212),
        "OMR": (12.9121, 80.2273),
        "Tambaram": (12.9229, 80.1275),
        "Adyar": (13.0067, 80.2570)
    },
    "Kolkata": {
        "Park Street": (22.5535, 88.3516),
        "Salt Lake": (22.5820, 88.4176),
        "Howrah": (22.5958, 88.2636),
        "New Town": (22.5826, 88.4626),
        "Ballygunge": (22.5326, 88.3643),
        "Jadavpur": (22.4986, 88.3732)
    },
    "Pune": {
        "Koregaon Park": (18.5362, 73.8958),
        "Hinjewadi": (18.5912, 73.7389),
        "Viman Nagar": (18.5679, 73.9143),
        "Kothrud": (18.5074, 73.8077),
        "Hadapsar": (18.5089, 73.9260),
        "Wakad": (18.5978, 73.7636)
    },
    "Ahmedabad": {
        "Navrangpura": (23.0358, 72.5639),
        "Satellite": (23.0258, 72.5150),
        "Maninagar": (22.9960, 72.6020),
        "Vastrapur": (23.0395, 72.5268),
        "Bopal": (23.0395, 72.4652),
        "Chandkheda": (23.1151, 72.6069)
    }
}

# City selection for AQI
aqi_city = st.sidebar.selectbox("Select City:", list(city_area_coords.keys()), key="aqi_city")

# Area selection based on city
aqi_area = st.sidebar.selectbox("Select Area:", list(city_area_coords[aqi_city].keys()), key="aqi_area")

if st.sidebar.button("Check AQI"):
    lat, lon = city_area_coords[aqi_city][aqi_area]
    result = get_air_quality(lat, lon)
    
    if "error" in result:
        st.sidebar.error(f"Error: {result['error']}")
    else:
        st.sidebar.success(f"🌍 Air Quality in {aqi_area}, {aqi_city}")
        st.sidebar.write(f"**AQI:** {result['aqi']}")
        st.sidebar.write(f"**Level:** {aqi_level(result['aqi'])}")
        st.sidebar.write(f"**PM2.5:** {result['pm2_5']} µg/m³")
        st.sidebar.write(f"**PM10:** {result['pm10']} µg/m³")

st.sidebar.markdown("---")

st.sidebar.header("About the Project")
st.sidebar.write("""
This app predicts housing prices with advanced features:
- Price Prediction & Analysis
- Air Quality Monitoring
- Commute Time Calculator
- Mortgage & EMI Calculator
- Price Negotiation Assistant
""")


# -------------------------------------------
# Main Content
# -------------------------------------------
st.markdown('<div class="overlay">', unsafe_allow_html=True)
st.title("🏠 Housing Price Prediction & Analysis")
st.write("Comprehensive real estate intelligence platform with price prediction, financial planning, and market insights!")

# Create tabs for different features
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Price Prediction", "🚗 Commute Calculator", "💰 Mortgage Calculator", "🤝 Negotiation Assistant"])

# -------------------------------------------
# TAB 1: Price Prediction
# -------------------------------------------
with tab1:
    st.header("Property Price Prediction")
    
    area = st.number_input("Enter area (sq ft):", min_value=500.0, max_value=10000.0, value=7420.0, step=100.0)
    bedrooms = st.number_input("Enter number of bedrooms:", min_value=1, max_value=10, value=4)
    bathrooms = st.number_input("Enter number of bathrooms:", min_value=1, max_value=5, value=2)
    stories = st.number_input("Enter number of stories:", min_value=1, max_value=5, value=3)

    # Prepare input for prediction
    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'stories': [stories]
    })
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    if st.button("Predict Price", key="predict_btn"):
        predicted_price = float(model.predict(input_data)[0])
        st.session_state.predicted_price = predicted_price
        
        st.balloons()
        st.success(f"🎉 Predicted Price: ${predicted_price:,.2f}")
        st.toast("Prediction done successfully 🏡")
        
        # ADDED: Distribution plots from basic version
        st.write("### 📊 Distribution of Training Data Features")
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.ravel()
        for i, col in enumerate(numerical_features + ['price']):
            data[col].hist(ax=axes[i], bins=15, edgecolor='black', color='orange')
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        plt.tight_layout()
        st.pyplot(fig)
        
        # ADDED: Predicted Price vs Area plot from basic version
        st.write("### 🧾 Predicted Price vs Area")
        fig2, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(data['area'], data['price'], color='blue', alpha=0.5, label='Training Data')
        ax.scatter(area, predicted_price, color='red', label='Predicted Price', s=200, marker='*', edgecolors='black', linewidths=2)
        ax.set_title('Predicted Price vs Area', fontsize=14, fontweight='bold')
        ax.set_xlabel('Area (sq ft)', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig2)
        
        col1, col2 = st.columns(2)
        with col1:
            listed_price = st.number_input(
                "Listed/Asking Price ($):",
                min_value=10000.0,
                value=float(st.session_state.get("predicted_price", 500000.0)),
                step=1000.0
            )
        with col2:
            property_condition = st.selectbox(
                "Property Condition",
                ["Excellent", "Good", "Fair", "Needs Renovation"]
            )

    # Property Recommendation System
    st.markdown("### 🏡 Property Recommendation System")
    rec_location = st.text_input("Enter your preferred location (city):")
    rec_budget = st.number_input("Enter your budget ($):", min_value=0, max_value=10000000, value=500000, step=10000)

    def recommend_properties(location, budget, top_k=4):
        if location not in location_map:
            return []

        loc_code = location_map[location]
        budget_range_low = budget * 0.7
        budget_range_high = budget * 1.3
        filtered = data[(data['price'] >= budget_range_low) & (data['price'] <= budget_range_high)]

        if filtered.empty:
            return []

        filtered = filtered.copy()
        filtered['price_norm'] = (filtered['price'] - filtered['price'].mean()) / filtered['price'].std()
        filtered['area_norm'] = (filtered['area'] - filtered['area'].mean()) / filtered['area'].std()

        input_vec = np.array([
            (budget - filtered['price'].mean()) / filtered['price'].std(),
            (area - filtered['area'].mean()) / filtered['area'].std(),
            loc_code
        ]).reshape(1, -1)

        prop_features = filtered[['price_norm', 'area_norm', 'city_code']].values
        similarities = cosine_similarity(input_vec, prop_features)[0]
        filtered['similarity'] = similarities
        filtered_sorted = filtered.sort_values(by='similarity', ascending=False)

        return filtered_sorted.head(top_k)

    if rec_location and rec_budget > 0:
        recommendations = recommend_properties(rec_location, rec_budget)
        if len(recommendations) == 0:
            st.info("No properties found matching the criteria.")
        else:
            st.write(f"Top {len(recommendations)} recommended properties near {rec_location} within budget.")
            cols = st.columns(len(recommendations))
            for idx, (_, row) in enumerate(recommendations.iterrows()):
                with cols[idx]:
                    st.markdown(f"**Location:** {row['city'] if 'city' in row else rec_location}")
                    st.markdown(f"**Price:** ${row['price']:,.2f}")
                    st.markdown(f"**Area:** {row['area']} sq ft")
                    st.markdown(f"**Bedrooms:** {row['bedrooms']}")
                    st.markdown(f"**Stories:** {row['stories']}")
                    st.markdown("---")

# -------------------------------------------
# TAB 2: Commute Calculator
# -------------------------------------------
with tab2:
    st.header("🚗 Commute Time Calculator")
    st.write("Calculate travel time from your property to major business districts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        origin_city = st.selectbox("Property Location (City):", 
                                   ["Hyderabad", "Delhi", "Mumbai", "Bangalore", "Chennai"],
                                   key="origin_city")
        origin_area = st.text_input("Area/Locality:", "Gachibowli", key="origin_area")
    
    with col2:
        destination = st.selectbox("Business District:", 
                                   ["Hitech City", "Connaught Place", "BKC", "Whitefield", "OMR"],
                                   key="destination")
        google_maps_key = st.text_input("Google Maps API Key (Optional):", 
                                       "YOUR_GOOGLE_MAPS_API_KEY", 
                                       type="password",
                                       help="Leave default for demo data")
    
    if st.button("Calculate Commute Time"):
        with st.spinner("Calculating route..."):
            origin = f"{origin_area}, {origin_city}"
            result = calculate_commute_time(origin, destination, google_maps_key)
            
            if result:
                st.success("✅ Route Calculated!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Travel Time", f"{result['duration']} mins", 
                             help="Estimated time with current traffic")
                
                with col2:
                    st.metric("Distance", f"{result['distance']} km",
                             help="Total distance to travel")
                
                with col3:
                    traffic_color = "🟢" if result['traffic'] == "Light" else "🟡" if result['traffic'] == "Moderate" else "🔴"
                    st.metric("Traffic", f"{traffic_color} {result['traffic']}",
                             help="Current traffic conditions")
                
                # Traffic insights
                st.info(f"""
                **Commute Insights:**
                - Average daily commute: ~{result['duration']*2} mins (round trip)
                - Monthly commute time: ~{result['duration']*2*22/60:.1f} hours
                - Fuel cost estimate (₹100/L): ₹{result['distance']*2*22*10:.0f}/month
                """)
                
                # Best times to travel
                st.write("### ⏰ Optimal Travel Times")
                times_df = pd.DataFrame({
                    "Time Slot": ["6:00 AM - 8:00 AM", "8:00 AM - 10:00 AM", "10:00 AM - 5:00 PM", 
                                 "5:00 PM - 8:00 PM", "8:00 PM - 10:00 PM"],
                    "Traffic Level": ["Light", "Heavy", "Moderate", "Very Heavy", "Light"],
                    "Est. Duration": [f"{result['duration']*0.7:.0f} mins", 
                                     f"{result['duration']*1.3:.0f} mins",
                                     f"{result['duration']:.0f} mins",
                                     f"{result['duration']*1.5:.0f} mins",
                                     f"{result['duration']*0.8:.0f} mins"]
                })
                st.dataframe(times_df, use_container_width=True)
            else:
                st.warning("⚠️ Using demo data. Add Google Maps API key for real-time data.")

# -------------------------------------------
# TAB 3: Mortgage Calculator
# -------------------------------------------
with tab3:
    st.header("💰 Mortgage & EMI Calculator")
    
    # Sub-tabs for different mortgage features
    subtab1, subtab2, subtab3 = st.tabs(["EMI Calculator", "Loan Eligibility", "Amortization Schedule"])
    
    with subtab1:
        st.subheader("Calculate Your Monthly EMI")
        
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amount = st.number_input("Loan Amount ($):", 
                                         min_value=10000, 
                                         max_value=10000000, 
                                         value=500000,
                                         step=10000)
            interest_rate = st.slider("Annual Interest Rate (%):", 
                                     min_value=5.0, 
                                     max_value=15.0, 
                                     value=8.5,
                                     step=0.1)
        
        with col2:
            loan_tenure = st.selectbox("Loan Tenure (Years):", 
                                      [5, 10, 15, 20, 25, 30],
                                      index=3)
            down_payment = st.number_input("Down Payment ($):", 
                                          min_value=0, 
                                          value=100000,
                                          step=10000)
        
        if st.button("Calculate EMI"):
            emi = calculate_emi(loan_amount, interest_rate, loan_tenure)
            total_amount = emi * loan_tenure * 12
            total_interest = total_amount - loan_amount
            
            st.success(f"### Monthly EMI: ${emi:,.2f}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Amount Payable", f"${total_amount:,.2f}")
            with col2:
                st.metric("Total Interest", f"${total_interest:,.2f}")
            with col3:
                st.metric("Principal Amount", f"${loan_amount:,.2f}")
            
            # Pie chart for breakdown
            fig = go.Figure(data=[go.Pie(
                labels=['Principal', 'Interest'],
                values=[loan_amount, total_interest],
                hole=.3,
                marker_colors=['#ff7f0e', '#1f77b4']
            )])
            fig.update_layout(title="Loan Breakdown", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Compare different interest rates
            st.write("### 📊 Compare Different Interest Rates")
            rates = [interest_rate - 1, interest_rate, interest_rate + 1, interest_rate + 2]
            emis = [calculate_emi(loan_amount, r, loan_tenure) for r in rates]
            
            comparison_df = pd.DataFrame({
                "Interest Rate (%)": rates,
                "Monthly EMI ($)": emis,
                "Total Interest ($)": [(emi * loan_tenure * 12 - loan_amount) for emi in emis]
            })
            st.dataframe(comparison_df, use_container_width=True)
    
    with subtab2:
        st.subheader("Check Your Loan Eligibility")
        
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_income = st.number_input("Monthly Income ($):", 
                                            min_value=1000,
                                            max_value=1000000,
                                            value=10000,
                                            step=500)
            existing_emi = st.number_input("Existing EMI ($):", 
                                          min_value=0,
                                          value=0,
                                          step=100)
        
        with col2:
            requested_loan = st.number_input("Requested Loan Amount ($):", 
                                            min_value=10000,
                                            max_value=10000000,
                                            value=400000,
                                            step=10000)
            eligibility_tenure = st.selectbox("Loan Tenure (Years):", 
                                             [5, 10, 15, 20, 25, 30],
                                             index=3,
                                             key="elig_tenure")
        
        eligibility_rate = st.slider("Expected Interest Rate (%):", 
                                     min_value=5.0,
                                     max_value=15.0,
                                     value=8.5,
                                     step=0.1,
                                     key="elig_rate")
        
        if st.button("Check Eligibility"):
            result = check_loan_eligibility(monthly_income, existing_emi, requested_loan, 
                                           eligibility_rate, eligibility_tenure)
            
            if result['eligible']:
                st.success("✅ Congratulations! You are eligible for this loan.")
            else:
                st.error("❌ You may not be eligible for the full loan amount.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("FOIR Ratio", f"{result['foir']}%", 
                         help="Fixed Obligation to Income Ratio (Should be < 50%)")
            with col2:
                st.metric("New EMI", f"${result['new_emi']:,.2f}")
            with col3:
                st.metric("Total EMI", f"${result['total_emi']:,.2f}")
            
            st.info(f"""
            **Eligibility Summary:**
            - Maximum Eligible Loan: ${result['max_eligible_loan']:,.2f}
            - Monthly Income: ${monthly_income:,.2f}
            - Existing Obligations: ${existing_emi:,.2f}
            - Available for New Loan: ${monthly_income * 0.5 - existing_emi:,.2f}
            """)
            
            # FOIR visualization
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = result['foir'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "FOIR Ratio (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 50], 'color': "yellow"},
                        {'range': [50, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with subtab3:
        st.subheader("Amortization Schedule")
        st.write("See how your loan gets paid off over time")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amort_principal = st.number_input("Loan Amount ($):", 
                                             min_value=10000,
                                             value=500000,
                                             step=10000,
                                             key="amort_principal")
        with col2:
            amort_rate = st.slider("Interest Rate (%):", 
                                  min_value=5.0,
                                  max_value=15.0,
                                  value=8.5,
                                  step=0.1,
                                  key="amort_rate")
        with col3:
            amort_tenure = st.selectbox("Tenure (Years):", 
                                       [5, 10, 15, 20, 25, 30],
                                       index=3,
                                       key="amort_tenure")
        
        if st.button("Generate Schedule"):
            schedule_df = generate_amortization_schedule(amort_principal, amort_rate, amort_tenure)
            
            # Summary metrics
            total_payment = schedule_df['EMI'].sum()
            total_interest = schedule_df['Interest'].sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Payment", f"${total_payment:,.2f}")
            with col2:
                st.metric("Total Interest", f"${total_interest:,.2f}")
            with col3:
                st.metric("Monthly EMI", f"${schedule_df['EMI'].iloc[0]:,.2f}")
            
            # Visualize principal vs interest over time
            yearly_data = schedule_df.groupby(schedule_df['Month'] // 12).agg({
                'Principal': 'sum',
                'Interest': 'sum'
            }).reset_index()
            yearly_data['Year'] = yearly_data['Month'] + 1
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Principal', x=yearly_data['Year'], y=yearly_data['Principal']))
            fig.add_trace(go.Bar(name='Interest', x=yearly_data['Year'], y=yearly_data['Interest']))
            fig.update_layout(
                title="Yearly Payment Breakdown (Principal vs Interest)",
                xaxis_title="Year",
                yaxis_title="Amount ($)",
                barmode='stack',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed schedule (first and last 12 months)
            st.write("### 📋 Detailed Amortization Schedule")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**First Year (Months 1-12)**")
                st.dataframe(schedule_df.head(12), use_container_width=True)
            
            with col2:
                st.write(f"**Last Year (Months {len(schedule_df)-11}-{len(schedule_df)})**")
                st.dataframe(schedule_df.tail(12), use_container_width=True)
            
            # Download option
            csv = schedule_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Schedule as CSV",
                data=csv,
                file_name=f"amortization_schedule_{amort_tenure}years.csv",
                mime="text/csv"
            )

# -------------------------------------------
# TAB 4: Negotiation Assistant
# -------------------------------------------
with tab4:
    st.header("🤝 Price Negotiation Assistant")
    st.write("Get data-driven insights to negotiate the best deal")
    
    # Market conditions selector
    market_condition = st.selectbox(
        "Current Market Condition:",
        ["Buyer's Market", "Normal", "Seller's Market"],
        index=1,
        help="Buyer's Market: High inventory, low demand | Seller's Market: Low inventory, high demand"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        listed_price_nego = st.number_input(
            "Listed/Asking Price ($):", 
            min_value=10000.0,
            value=float(st.session_state.get('predicted_price', 500000.0)),
            step=1000.0,
            key="listed_price_nego"
        )
    
    with col2:
        property_condition = st.selectbox("Property Condition:",
                                         ["Excellent", "Good", "Fair", "Needs Renovation"])
    
    days_on_market = st.slider("Days on Market:", 
                               min_value=0,
                               max_value=365,
                               value=30,
                               help="How long has the property been listed?")
    
    if st.button("Analyze Negotiation Strategy"):
        
        # Calculate negotiation range
        negotiation = calculate_negotiation_range(listed_price_nego, market_condition)
        market_timing = analyze_market_timing()
        
        st.success("✅ Negotiation Strategy Generated!")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Listed Price", f"${listed_price_nego:,.2f}")
            st.metric("Market Condition", market_condition)
        
        with col2:
            st.metric("Optimal Offer", f"${negotiation['optimal_offer']:,.2f}",
                     delta=f"-${negotiation['negotiation_room']:,.2f}")
            st.metric("Market Timing", market_timing['timing'])
        
        with col3:
            st.metric("Negotiation Room", f"{negotiation['percentage_below']}%",
                     help="Percentage below asking price")
            st.metric("Days on Market", days_on_market)
        
        # Negotiation range visualization
        st.write("### 💡 Recommended Offer Range")
        
        fig = go.Figure()
        
        # Add bars for different offer levels
        offers = ['Minimum Offer', 'Optimal Offer', 'Maximum Offer', 'Listed Price']
        values = [negotiation['min_offer'], negotiation['optimal_offer'], 
                 negotiation['max_offer'], listed_price_nego]
        colors = ['red', 'green', 'orange', 'blue']
        
        fig.add_trace(go.Bar(
            x=offers,
            y=values,
            marker_color=colors,
            text=[f"${v:,.0f}" for v in values],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Negotiation Price Ladder",
            yaxis_title="Price ($)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed strategy
        st.write("### 📝 Negotiation Strategy & Tips")
        
        # Adjust strategy based on days on market
        if days_on_market < 7:
            urgency = "High"
            strategy_tip = "Property is fresh. Be prepared to act quickly but don't overpay."
        elif days_on_market < 30:
            urgency = "Moderate"
            strategy_tip = "Standard timeline. You have room to negotiate but don't lowball."
        elif days_on_market < 90:
            urgency = "Low"
            strategy_tip = "Property has been on market for a while. Seller may be motivated. Negotiate firmly."
        else:
            urgency = "Very Low"
            strategy_tip = "Property is stale. Seller is likely very motivated. You have strong negotiating power."
        
        st.info(f"""
        **Your Negotiation Position: {urgency} Urgency**
        
        {strategy_tip}
        
        **Recommended Opening Offer:** ${negotiation['optimal_offer']:,.2f} ({negotiation['percentage_below']}% below asking)
        
        **Your Negotiation Range:**
        - 🔴 Minimum: ${negotiation['min_offer']:,.2f} (Walk away if above this)
        - 🟢 Target: ${negotiation['optimal_offer']:,.2f} (Aim for this price)
        - 🟠 Maximum: ${negotiation['max_offer']:,.2f} (Don't exceed this)
        
        **Market Timing:** {market_timing['timing']} - {market_timing['advice']}
        """)
        
        # Property-specific adjustments
        condition_adjustments = {
            "Excellent": 0,
            "Good": -2,
            "Fair": -5,
            "Needs Renovation": -10
        }
        
        adjustment = condition_adjustments[property_condition]
        adjusted_offer = negotiation['optimal_offer'] * (1 + adjustment/100)
        
        st.write("### 🔧 Property Condition Adjustment")
        if adjustment < 0:
            st.warning(f"""
            Based on '{property_condition}' condition, consider adjusting your offer down by {abs(adjustment)}%.
            
            **Condition-Adjusted Optimal Offer:** ${adjusted_offer:,.2f}
            
            Use the condition as a negotiation point to justify a lower offer.
            """)
        else:
            st.success("Property is in excellent condition. Your calculated offer stands.")
        
        # Negotiation tactics
        st.write("### 🎯 Proven Negotiation Tactics")
        
        tactics_df = pd.DataFrame({
            "Tactic": [
                "Start Low (But Reasonable)",
                "Point Out Needed Repairs",
                "Mention Comparable Sales",
                "Show Pre-Approval Letter",
                "Be Ready to Walk Away",
                "Limit Contingencies"
            ],
            "Effectiveness": ["High", "Medium", "High", "Medium", "High", "Medium"],
            "Risk Level": ["Medium", "Low", "Low", "Low", "High", "Medium"]
        })
        
        st.dataframe(tactics_df, use_container_width=True)
        
        # Comparable properties analysis
        st.write("### 🏘️ Comparable Properties Analysis")
        
        # Find similar properties from the dataset
        similar_props = data[
            (data['area'] >= area * 0.9) & 
            (data['area'] <= area * 1.1) &
            (data['bedrooms'] == bedrooms)
        ].head(5)
        
        if not similar_props.empty:
            avg_price = similar_props['price'].mean()
            price_range = (similar_props['price'].min(), similar_props['price'].max())
            
            st.write(f"""
            **Market Comparison:**
            - Average price for similar properties: ${avg_price:,.2f}
            - Price range: ${price_range[0]:,.2f} - ${price_range[1]:,.2f}
            - Your listed price vs market: {((listed_price_nego/avg_price - 1) * 100):+.1f}%
            """)
            
            if listed_price_nego > avg_price * 1.1:
                st.warning(f"⚠️ Listed price is {((listed_price_nego/avg_price - 1) * 100):.1f}% above market average. Strong negotiation leverage!")
            elif listed_price_nego < avg_price * 0.9:
                st.info(f"💰 Listed price is below market average. This could be a good deal!")
            else:
                st.success("✅ Listed price is in line with market averages.")


# -------------------------------------------
# AI Chat Assistant Section
# -------------------------------------------
st.markdown("---")
st.markdown("### 💬 AI Real Estate Assistant")
st.write("Ask questions about housing, prices, mortgages, or negotiation strategies!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask your real estate query...", key="chat_input_unique")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Check if query is about air quality
    if "air quality" in user_input.lower() or "aqi" in user_input.lower():
        city_area_coords_lower = {
            "hyderabad": {"default": (17.3850, 78.4867)},
            "delhi": {"default": (28.6139, 77.2090)},
            "mumbai": {"default": (19.0760, 72.8777)},
            "bangalore": {"default": (12.9716, 77.5946)},
            "chennai": {"default": (13.0827, 80.2707)},
            "kolkata": {"default": (22.5726, 88.3639)},
            "pune": {"default": (18.5204, 73.8567)},
            "ahmedabad": {"default": (23.0225, 72.5714)}
        }
        
        detected_city = "hyderabad"
        location_name = "Hyderabad"
        
        query_lower = user_input.lower()
        for city_name in city_area_coords_lower.keys():
            if city_name in query_lower:
                detected_city = city_name
                location_name = city_name.capitalize()
                break
        
        lat, lon = city_area_coords_lower[detected_city]["default"]
        result = get_air_quality(lat, lon)
        
        if "error" not in result:
            ai_reply = f"Current Air Quality in {location_name}:\n\n"
            ai_reply += f"• **AQI:** {result['aqi']} - {aqi_level(result['aqi'])}\n"
            ai_reply += f"• **PM2.5:** {result['pm2_5']} µg/m³\n"
            ai_reply += f"• **PM10:** {result['pm10']} µg/m³\n\n"
            ai_reply += "The Air Quality Index (AQI) tells you how clean or polluted your air is. Lower values are better!"
        else:
            ai_reply = f"Sorry, I couldn't fetch air quality data for {location_name}. Error: {result['error']}"
        
        with st.chat_message("assistant"):
            st.markdown(ai_reply)
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
    else:
        # Regular AI assistant response
        generate_desc_trigger = any(keyword in user_input.lower() for keyword in ["describe", "description", "generate description", "property description"])

        system_prompt = """You are a professional real estate assistant with expertise in:
        - Property valuation and pricing
        - Mortgage and financing options
        - Negotiation strategies
        - Market analysis and trends
        - Air quality and environmental factors
        
        Give short, practical, and actionable advice."""

        if generate_desc_trigger:
            system_prompt += " Also, generate a concise and appealing property description based on the user's input features."

        headers = {"Authorization": "Bearer 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"}
        api_url = "https://api.groq.com/openai/v1/chat/completions"

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        }

        with st.spinner("💭 AI assistant is thinking..."):
            try:
                response = requests.post(api_url, headers=headers, json=payload)
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    ai_reply = result["choices"][0]["message"]["content"]
                elif "error" in result:
                    ai_reply = f"⚠️ API Error: {result['error'].get('message', 'Unknown error occurred.')}"
                else:
                    ai_reply = "⚠️ Unexpected API response: unable to fetch reply."
            except Exception as e:
                ai_reply = f"❌ Request failed: {e}"

        with st.chat_message("assistant"):
            st.markdown(ai_reply)
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
