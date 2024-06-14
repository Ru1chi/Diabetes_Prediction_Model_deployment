# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:53:31 2024

@author: sr322
"""

import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open(r'trained_model.sav','rb'))#load function is used to load the saved model


#creating a function for prediction
def diabetes_prediction(input_data):
     
 

     # changing the input_data to numpy array
     input_data_as_numpy_array = np.asarray(input_data)

     # reshape the array as we are predicting for one instance
     input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

     prediction = loaded_model.predict(input_data_reshaped)
     print(prediction)

     if (prediction[0] == 0):
       return  'The person is not diabetic'
     else:
       return  'The person is diabetic'


#main function
def main():
    
    #giving the title
    st.title('Diabete Prediction Web App')
    
    
    
    #getting the input_data from the user
    
    Pregnancies=st.number_input("Number of Pregnencies-")
    Glucose=st.number_input("Glucose level-")
    BloodPressure=st.number_input("Blodd Pressure level-")
    SkinThickness=st.number_input("Skinthikness level-")
    Insulin=st.number_input("Insulin level-")
    BMI=st.number_input("BMI rate is-")
    DiabetesPedigreeFunction=st.number_input("DiabetesPedigreeFunction-")
    Age=st.number_input("Ageof the person-")
    
    
    #code for prediction
    diagnosis=''#null string will show the final result    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    
    st.success(diagnosis)
    
    
    



if __name__ == '__main__':
    main()
   
