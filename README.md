# SymptoForecast: Intelligent Healthcare Solutions

SymptoForecast is a Streamlit web application designed to provide disease predictions based on symptom inputs. This application offers predictions for various medical conditions including diabetes, heart disease, and Parkinson's disease. By leveraging machine learning models, users can input symptoms and receive predictions about the likelihood of having a particular medical condition.

## Table of Contents

- [Challenges Faced](#challenges-faced)
- [Available Features](#available-features)


CHALLENGES FACED:

Building the SymptoForecast application came with its fair share of challenges:

Feature Mismatch Issue: A notable challenge we encountered was the feature mismatch between the symptom inputs selected by the user and the number of features expected by the machine learning models. This led to issues with the Standard Scaler in the model, which was expecting a fixed number of features for scaling.

Integration Issues: Integrating the machine learning models with the Streamlit application also posed challenges. We had to learn about various tools and techniques to properly serialize and load the models using libraries like joblib and pickle.

Data Preprocessing Challenges: Preprocessing symptom data and mapping it to the correct format for prediction was another hurdle. Ensuring that the symptom inputs were transformed and structured appropriately for the model was a complex task.

Available Features
Disease Prediction: This section allows you to input a range of symptoms to predict the likelihood of having a particular disease. The symptoms are selected using checkboxes, and the application displays the prediction result upon clicking the "Predict" button.

Diabetes Prediction: Predicts the likelihood of having diabetes based on input features such as number of pregnancies, glucose level, blood pressure, etc.

Heart Disease Prediction: Predicts the likelihood of having heart disease based on input features like age, sex, chest pain types, and various medical measurements.

Parkinson's Prediction: Predicts the likelihood of having Parkinson's disease based on input features including vocal frequency and other medical measurements.

**WE ARE YET TO DEPLOY THIS MACHINE LEARNING WEB APPLICATION.**
