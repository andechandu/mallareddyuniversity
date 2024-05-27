import streamlit as st
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load('terminal (2).pkl')
    if model is None:
        st.error("Model file is empty.")
except FileNotFoundError:
    st.error("Model file not found. Please ensure that 'terminal (2)' exists in the current directory.")
except Exception as e:
    st.error(f"Error loading model: {e}")

def predict_delivery_time(Delivery_person_Age,distance,multiple_deliveries,Delivery_person_Ratings,Festival,City):
    try:
        # Create input array
        features = np.array([Delivery_person_Age,distance,multiple_deliveries,Delivery_person_Ratings,Festival,City]).reshape(1, -1)

        # Predict
        prediction = model.predict(features)

        return prediction[0]  # Adjust this according to your model's output shape
    except Exception as e:
        st.error(f"Prediction error: {e}")



def main():
    st.title('Food Delivery Time Prediction')
    st.write('Fill out the form below to predict delivery time.')
    # Input form
    with st.form(key='delivery_time_form'):
        Delivery_person_Age = st.number_input('Delivery Person Age', min_value=18, step=1)
        Delivery_person_Ratings = st.slider('Delivery Person Ratings', min_value=1.0, max_value=5.0, step=0.1)
        distance = st.number_input('Distance', min_value=1, step=1)
        multiple_deliveries = st.number_input('Multiple Deliveries', min_value=0.0,max_value=5.0, step=1.0)
        Festival = st.number_input('Festival', min_value=0,max_value=1, step=1)
        City = st.number_input('City', min_value=1,max_value=2, step=1)



        
        

        if st.form_submit_button(label='Predict'):
            prediction = predict_delivery_time(Delivery_person_Age,distance,multiple_deliveries,Delivery_person_Ratings,Festival,City)
            if prediction is not None:
                st.write(f'Predicted Delivery Time: {prediction} minutes')

if __name__ == '__main__':
    main()
