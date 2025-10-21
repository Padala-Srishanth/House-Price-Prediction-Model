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

# Prepare location encoding (simple for demo)
# Map city names to numeric codes for similarity
location_map = {loc: idx for idx, loc in enumerate(data['city'].unique())} if 'city' in data.columns else {}
if 'city' in data.columns:
    data['city_code'] = data['city'].map(location_map)
else:
    # if no city column, create a dummy one with 0
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

st.sidebar.header("About the Project")
st.sidebar.write("""
This app predicts housing prices based on property details,
provides AI insights, and recommends properties based on location and budget.
""")


# -------------------------------------------
# Main Content
# -------------------------------------------
st.markdown('<div class="overlay">', unsafe_allow_html=True)
st.title("🏠 Housing Price Prediction App")
st.write("Enter property details below and interact with our AI real estate assistant!")


area = st.number_input("Enter area (sq ft):", min_value=500, max_value=10000, value=7420, step=100)
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


# -------------------------------------------
# Price Prediction Button
# -------------------------------------------
if st.button("Predict Price"):
    predicted_price = model.predict(input_data)[0]

    st.balloons()
    st.success(f"🎉 Predicted Price: ${predicted_price:,.2f}")
    st.toast("Prediction done successfully 🏡")

    st.write("### 📊 Distribution of Training Data Features")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    for i, col in enumerate(numerical_features + ['price']):
        data[col].hist(ax=axes[i], bins=15, edgecolor='black', color='orange')
        axes[i].set_title(f'{col} Distribution')
    plt.tight_layout()
    st.pyplot(fig)

    st.write("### 🧾 Predicted Price vs Area")
    plt.figure(figsize=(8, 6))
    plt.scatter(data['area'], data['price'], color='blue', alpha=0.5, label='Training Data')
    plt.scatter(area, predicted_price, color='red', label='Predicted Price', s=100)
    plt.title('Predicted Price vs Area')
    plt.xlabel('Area (sq ft)')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid()
    st.pyplot(plt)


# -------------------------------------------
# Property Recommendation System
# -------------------------------------------
st.markdown("### 🏡 Property Recommendation System")
rec_location = st.text_input("Enter your preferred location (city):")
rec_budget = st.number_input("Enter your budget ($):", min_value=0, max_value=10000000, value=500000, step=10000)

def recommend_properties(location, budget, top_k=4):
    if location not in location_map:
        return []

    loc_code = location_map[location]

    # Filter dataset around budget ±30%
    budget_range_low = budget * 0.7
    budget_range_high = budget * 1.3
    filtered = data[(data['price'] >= budget_range_low) & (data['price'] <= budget_range_high)]

    if filtered.empty:
        return []

    # Features for similarity: normalized price, area, city_code
    filtered = filtered.copy()

    # Normalize price and area for similarity
    filtered['price_norm'] = (filtered['price'] - filtered['price'].mean()) / filtered['price'].std()
    filtered['area_norm'] = (filtered['area'] - filtered['area'].mean()) / filtered['area'].std()
    # city code does not normalize since categorical numeric

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
# AI Chat Assistant Section (With Description Generator)
# -------------------------------------------
st.markdown("### 💬 AI Real Estate Assistant")
st.write("Ask questions like:\n- Why is my house price lower?\n- What features increase home value?\n- You can also ask to generate a property description.")


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Ask your real estate query...", key="chat_input_unique")


if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Detect if user wants property description
    generate_desc_trigger = any(keyword in user_input.lower() for keyword in ["describe", "description", "generate description", "property description"])

    # Build system prompt dynamically
    system_prompt = "You are a professional real estate assistant. Give short, useful housing insights."

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
