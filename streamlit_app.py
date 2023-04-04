import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open('E:/Data Science Projects/Project Deployment/trained_model.sav', 'rb'))

# creating a function for prediction

def diabetes_prediction(input_data):
    input_data = (5,166,72,19,175,25.8,0.587,51)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the input data
    # std_data = loaded_model.transform(input_data_reshaped)
    # print(std_data)

    prediction = loaded_model.predict(input_data_reshaped)
    # print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
def main():
    st.title('Diabetes Prediction model')
    
    #getting the data from the user


    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Enter your Glucose Level')
    BloodPressure = st.text_input('Enter your Blood Pressure ')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Value')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Enter Age')

    #code for prediction
    diagnosis = ''

    #creating a button for Prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)


if __name__ == '__main__':
    main()