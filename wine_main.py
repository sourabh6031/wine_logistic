import streamlit as st
import pandas as pd
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("wine.csv")
st.title("Let's make a model for the wine dataset")
st.subheader("Sample Dataframe for references")

st.dataframe(df.iloc[:,1:-1]) 

try:
    with open('wine_logreg_model.pkl','rb') as file:
        model = pickle.load(file)

    with open("selected_features.pkl",'rb') as feats:
        selected_features = pickle.load(feats)

    with open('scaler.pkl','rb') as std_scaler:
        scaler = pickle.load(std_scaler)

except Exception as e:
    print(f"Error loading model: {e}")

st.divider()

# Function to preprocess user input
def preprocess_input(user_input):
    """
    user_input: Dictionary containing feature values
    """
    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])

    # Keep only selected features
    input_df = input_df[selected_features]

    # Apply scaling
    scaled_input = scaler.transform(input_df)

    return scaled_input

# TAKING INPUT FROM USER    
user_input = {}
for feature in selected_features:
    user_input[feature] = st.number_input(f"Enter values for - {feature}", value=0.0)

if st.button("Predict"):
    processed_input = preprocess_input(user_input)
    prediction = model.predict(processed_input)
    st.write("Here is my prediction...")
    
    if prediction[0] == 0:
        #st.write(f"Prediction: {prediction[0]}")
        st.subheader("It's Class 1, First Region")
    elif prediction[0] == 1:
        #st.write(f"Prediction: {prediction[0]}")
        st.subheader("It's Class 2, Second Region")
    else:
        #st.write(f"Prediction: {prediction[0]}")
        st.subheader("It's Class 3, Third Region")
    
    st.balloons()