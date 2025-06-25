import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# -------------------- Data Loading & Preprocessing --------------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv(r"C:\\Users\\tariq\\OneDrive\\Desktop\\Summer_Projects\\Human_Resources_Department_Solution\\Human_Resources.csv")
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
    # Categorical and numerical features
    cat_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
    num_features = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'OverTime']
    X_cat = df[cat_features]
    X_num = df[num_features]
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False)
    X_cat_encoded = encoder.fit_transform(X_cat)
    X_all = np.hstack([X_cat_encoded, X_num])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_all)
    y = df['Attrition']
    model = LogisticRegression()
    model.fit(X_scaled, y)
    return model, scaler, encoder, cat_features, num_features

model, scaler, encoder, cat_features, num_features = load_and_train()

# -------------------- Streamlit UI --------------------
st.title("HR Attrition Prediction (Logistic Regression Demo)")
st.write("Enter employee details to predict attrition risk.")

# Widgets for categorical features
user_cat = []
for col in cat_features:
    options = encoder.categories_[cat_features.index(col)]
    user_cat.append(st.selectbox(col, options))

# Widgets for numerical features
user_num = []
user_num.append(st.slider("Age", 18, 60, 30))
user_num.append(st.slider("Distance From Home (km)", 1, 30, 5))
user_num.append(st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000, step=100))
user_num.append(1 if st.selectbox("OverTime", ["No", "Yes"]) == "Yes" else 0)

# Prepare input for prediction
user_cat_encoded = encoder.transform([user_cat])
user_cat_encoded = np.asarray(user_cat_encoded)
user_num_array = np.array(user_num).reshape(1, -1)
user_features = np.hstack([user_cat_encoded, user_num_array])
user_features_scaled = scaler.transform(user_features)

if st.button("Predict Attrition"):
    pred = model.predict(user_features_scaled)[0]
    prob = model.predict_proba(user_features_scaled)[0][1]
    st.markdown("---")
    st.subheader("Prediction Result:")
    st.write(f"**Attrition:** {'Yes' if pred == 1 else 'No'}")
    st.write(f"**Probability of Attrition:** {prob:.2%}")

# For future: add more features, better layout, explanations, etc. 