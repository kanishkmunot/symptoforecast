import pickle
import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from PIL import Image

# loading the saved models
diabetes_model = pickle.load(open("C:/Users/kmuno/OneDrive/Desktop/Hackathon/diabetes_model.sav", 'rb'))
heart_disease_model = pickle.load(open("C:/Users/kmuno/OneDrive/Desktop/Hackathon/heart_disease_model.sav", 'rb'))
parkinsons_model = pickle.load(open("C:/Users/kmuno/OneDrive/Desktop/Hackathon/parkinsons_model.sav", 'rb'))
pipeline = joblib.load('model_pipeline.joblib')

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('SymptoForecast',
                      ['Home', 'Disease Prediction', 'Diabetes Prediction',
                       'Heart Disease Prediction', 'Parkinsons Prediction'],
                      icons=['activity', 'heart', 'person'],
                      default_index=0)

# Home Page
if selected == 'Home':
    st.title('Revolutionizing Healthcare with ML Solutions')

    col1, col2 = st.columns(2, gap="large")

    with col1:
        image = Image.open('hero.jpg')
        st.image(image)
    with col2:
        st.subheader("Transforming Care with AI Innovation: Explore Intelligent Healthcare Solutions that Define a New Era in Wellness.")

# Disease Prediction Page
symptom_names = [' acidity', 'pain', 'discomfort', ' breathlessness', 'micturition', 'pain', ' chills', ' constipation', 'sneezing', ' cough', ' cramps', ' fatigue', ' headache', 'fever', ' indigestion', 'pain', 'swings', 'wasting', 'weakness', 'pain', 'movements', 'throat', 'pimples', ' shivering', 'rash', 'neck', 'pain', 'eyes', ' vomiting', 'limbs', 'gain', 'loss', 'skin', 'itching', 'pain', ' acidity', ' anxiety', ' blackheads', 'discomfort', ' blister', ' breathlessness', ' bruising', 'pain', ' chills', 'feets', ' cough', ' cramps', ' dehydration', ' dizziness', ' fatigue', 'of urine', ' headache', 'fever', ' indigestion', 'pain', 'pain', ' lethargy', 'appetite', 'swings', ' nausea', 'pain', 'eruptions', 'movements', 'region', 'throat', 'pimples', ' restlessness', ' shivering', 'peeling', 'rash', 'neck', 'pain', 'eyes', ' sweating', 'joints', 'tongue', ' vomiting', 'limbs', 'side', 'gain', 'loss', 'skin', 'pain', 'sensorium', ' anxiety', ' blackheads', ' blister', 'stool', 'vision', ' breathlessness', ' bruising', 'micturition', 'pain', ' chills', 'feets', 'urine', ' cough', 'urine', ' dehydration', ' diarrhoea', 'patches', ' dizziness', 'contacts', ' fatigue', 'of urine', ' headache', 'fever', 'pain', 'pain', 'pain', ' lethargy', 'appetite', 'balance', 'swings', 'stiffness', ' nausea', 'pain', 'eruptions', ' obesity', 'region', 'nose', ' restlessness', ' scurring', 'dusting', 'peeling', 'movements', 'pain', ' sweating', 'joints', 'stomach', 'tongue', ' vomiting', 'eyes', 'side', 'loss', 'skin', 'pain', 'sensorium', 'stool', 'vision', ' breathlessness', 'micturition', 'pain', 'urine', ' cough', 'urine', ' diarrhoea', 'patches', 'abdomen', ' dizziness', 'hunger', 'contacts', 'history', ' fatigue', ' headache', 'fever', 'pain', 'level', 'anus', 'concentration', ' lethargy', 'appetite', 'balance', 'swings', 'stiffness', ' nausea', ' obesity', 'walking', 'gases', 'nose', ' restlessness', ' scurring', 'dusting', 'nails', 'movements', ' urination', ' sweating', 'joints', 'stomach', 'legs', ' vomiting', 'eyes', 'loss', 'ooze', 'eyes', 'skin', 'pain', 'vision', ' breathlessness', 'pain', ' cough', 'urine', ' diarrhoea', 'abdomen', ' dizziness', 'hunger', 'history', ' fatigue', ' headache', 'fever', 'consumption', 'nails', 'itching', 'level', 'anus', 'concentration', ' lethargy', 'appetite', 'balance', 'sputum', ' nausea', 'walking', 'gases', 'nails', ' urination', 'neck', ' sweating', 'joints', 'vessels', 'legs', ' unsteadiness', 'ooze', 'eyes', 'skin', 'pain', 'vision', ' breathlessness', 'pain', ' constipation', 'urine', ' depression', ' diarrhoea', ' dizziness', 'history', 'rate', 'overload', ' headache', 'fever', 'consumption', 'nails', 'itching', 'appetite', ' malaise', 'sputum', ' nausea', ' obesity', 'walking', 'calf', 'eyes', 'neck', ' sweating', 'nodes', 'vessels', ' unsteadiness', 'eyes', 'skin', 'pain', 'vision', ' breathlessness', ' constipation', 'urine', ' depression', ' diarrhoea', 'thyroid', 'hunger', 'rate', 'overload', ' headache', ' irritability', 'appetite', ' malaise', 'fever', 'pain', ' nausea', ' obesity', ' phlegm', 'calf', 'eyes', ' sweating', 'nodes', 'urine', 'eyes', 'pain', 'nails', 'pain', ' diarrhoea', 'lips', 'thyroid', 'hunger', 'appetite', ' irritability', 'appetite', ' malaise', 'fever', 'pain', 'weakness', ' nausea', ' phlegm', ' sweating', 'nodes', 'disturbances', 'urine', 'eyes', 'pain', 'nails', 'pain', ' diarrhoea', 'lips', 'rate', 'appetite', ' irritability', 'appetite', ' malaise', 'fever', 'weakness', 'eyes', ' phlegm', ' polyuria', 'speech', 'nodes', 'extremeties', 'irritation', '(typhos)', 'disturbances', 'eyes', 'menstruation', 'failure', 'pain', 'pain', ' depression', 'rate', ' irritability', ' malaise', 'fever', 'pain', 'eyes', ' polyuria', 'transfusion', 'body', 'eyes', 'sputum', 'speech', 'extremeties', 'irritation', '(typhos)', 'eyes', 'menstruation', 'failure', 'pain', 'pain', ' coma', ' depression', ' irritability', ' malaise', 'pain', ' palpitations', 'transfusion', 'injections', 'body', 'eyes', 'sputum', 'pressure', 'nodes', 'eyes', 'menstruation', ' coma', ' irritability', ' malaise', 'pain', ' palpitations', 'injections', 'nose', 'pressure', 'bleeding', 'nodes', 'menstruation', ' congestion', ' malaise', 'pain', ' phlegm', 'body', 'nose', 'bleeding', 'pain', ' congestion', ' phlegm', 'body', 'sputum', 'pain', 'smell', 'sputum', 'smell', 'pain', 'pain']

if selected == 'Disease Prediction':
    st.title("Input your symptoms here")
    # Dropdown menu to select features
    selected_symptoms = st.multiselect('Select Features:', symptom_names)

    # Create an array to hold symptom values
    symptoms = []

    # Iterate through all symptom names
    for i, symptom_name in enumerate(symptom_names):
        unique_key = f'symptom_{i}'  # Generate a unique key for each input widget

        # If symptom_name is in the selected_symptoms list, create a number_input field
        if symptom_name in selected_symptoms:
            symptom_value = st.number_input(symptom_name.capitalize() + ':', key=unique_key, value=1, min_value=0, max_value=1, step=1)
        else:
            # If not selected, set the value to 0
            symptom_value = 0

        symptoms.append(symptom_value)

    # Button to make predictions
    if st.button('Predict'):
        # Prepare input data for prediction
        input_data = np.array([symptoms])  # Create a NumPy array

        # Make predictions using the pipeline
        prediction = pipeline.predict(input_data)

        # Display the prediction
        st.write(f'Prediction: {prediction[0]}')

# Other Disease Prediction Pages (Diabetes, Heart Disease, Parkinson's)
# ... (Your code for these sections goes here)


# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)




# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)













