import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
import pickle
import warnings
import numpy as np
import pandas as pd
from io import StringIO
import requests

# Load the Model
maternal_model = pickle.load(open('Model/best_model_gradient_boosting_classifier.sav', 'rb'))
fetal_model= pickle.load(open('Model/fetal_health_classifier.sav', 'rb'))

# Load models and data for Disease Prediction
with open("Model/kmeans_model.sav", "rb") as f:
    kmeans_model = pickle.load(f)

with open("Model/tfidf_vectorizer.sav", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("Model/symptom_data_with_clusters.sav", "rb") as f:
    disease_df = pickle.load(f)

# Constructing the sidebar for navigation

with st.sidebar:
    st.title("MediPredict : AI Health Analyzer")
    st.write("Welcome to MediPredict")
    st.write("Choose an option from the menu below to get started: ")

    selected = option_menu('MediPredict',['About Us', 'Pregnancy Risk Prediction','Fetal Health Prediction','Disease Prediction System'],
                           icons=['chat-square-text','hospital','capsule-pill', 'clipboard-data'],
                           default_index=0)
if (selected == 'About Us'):
    st.title("Welcome to MedPredict")
    st.write(
        "At MedPredict, our mission is to revolutionize healthcare by offering innovative solutions through predictive analysis. "
        "Our platform is specifically designed to address the intricate aspects of maternal and fetal health, providing accurate "
        "predictions and proactive risk management.")

    col1, col2 = st.columns(2)
    with col1:
        # Section 1: Pregnancy Risk Prediction
        st.header("1. Pregnancy Risk Prediction")
        st.write(
            "Our Pregnancy Risk Prediction feature utilizes advanced algorithms to analyze various parameters, including age, "
            "body sugar levels, blood pressure, and more. By processing this information, we provide accurate predictions of "
            "potential risks during pregnancy.")
        # Add an image for Pregnancy Risk Prediction
        st.image("Graphic_images/pregnancy_risk_image.jpg", caption="Pregnancy Risk Prediction", use_container_width=True)
    with col2:
        # Section 2: Fetal Health Prediction
        st.header("2. Fetal Health Prediction")
        st.write(
            "Fetal Health Prediction is a crucial aspect of our system. We leverage cutting-edge technology to assess the "
            "health status of the fetus. Through a comprehensive analysis of factors such as ultrasound data, maternal health, "
            "and genetic factors, we deliver insights into the well-being of the unborn child.")
        # Add an image for Fetal Health Prediction
        st.image("Graphic_images/fetal_health_image.jpg", caption="Fetal Health Prediction", use_container_width=True)

    # Section 3: Dashboard
    st.header("3. Disease Prediction System")
    st.write(
        "Our Dashboard provides a user-friendly interface for monitoring and managing health data. It offers a holistic "
        "view of predictive analyses, allowing healthcare professionals and users to make informed decisions. The Dashboard "
        "is designed for ease of use and accessibility.")
    # Add an image for Fetal Health Prediction
    st.image("Graphic_images/diseases.png", caption="Contagious Disease Prediction", use_container_width=True)

    # Closing note
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)

    st.write(
        "Thank you for choosing MediPredict. We are committed to advancing healthcare through technology and predictive analytics. "
        "Feel free to explore our features and take advantage of the insights we provide.")
    st.markdown("<br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
    st.write("Created and Constructed By R.Gajendran")

if(selected == 'Pregnancy Risk Prediction'):

    st.title("Pregnancy Risk Prediction")
    content = "Predicting the risk in pregnancy involves analyzing several parameters, including age, blood sugar levels, blood pressure, and other relevant factors"
    st.markdown(f"<div style='white-space: pre-wrap;'<b>{content}</b></div></br>",unsafe_allow_html=True)

    # Getting the input data from the user
    col1, col2, col3 = st.columns(3) # in a row how many columns

    with col1:
        age= st.text_input("Age of the Person", key="age")

    with col2:
        diastolicBP = st.text_input("Diastolic BP in mmHg")

    with col3:
        BS = st.text_input("Blood Glucose in mmol/L")

    with col1:
        bodyTemp = st.text_input("Body Temperature in Celsius")

    with col2:
        heartRate = st.text_input("Heart rate in beats per minute")
    riskLevel = ""
    predicted_risk = [0]
    # creating a button for Prediction
    with col1:
        if st.button('Predict Pregnancy Risk'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predicted_risk = maternal_model.predict([[age, diastolicBP, BS, bodyTemp, heartRate]])
            # st
            st.subheader("Risk Level:")
            if predicted_risk[0] == 0:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: green;">Low Risk</p></bold>',
                            unsafe_allow_html=True)
            elif predicted_risk[0] == 1:
                st.markdown(
                    '<bold><p style="font-weight: bold; font-size: 20px; color: orange;">Medium Risk</p></Bold>',
                    unsafe_allow_html=True)
            else:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: red;">High Risk</p><bold>',
                            unsafe_allow_html=True)
    with col2:
        if st.button("Clear"):
            st.rerun()
################################################################################################################################
if (selected == 'Disease Prediction System'):
    st.header("Disease Prediction Based on Symptoms")
    st.markdown(
        "This feature uses unsupervised learning (KMeans) and NLP techniques to find the most likely disease based on your symptoms.")

    user_symptoms = st.text_area("Enter your symptoms (e.g. fever, cough, fatigue):")

    if st.button("Predict Disease"):
        try:
            user_vec = tfidf_vectorizer.transform([user_symptoms]).toarray()
            cluster = kmeans_model.predict(user_vec)[0]

            cluster_df = disease_df[disease_df["Cluster"] == cluster]
            cluster_symptoms_vec = tfidf_vectorizer.transform(cluster_df["symptoms"]).toarray()

            similarities = cosine_similarity(user_vec, cluster_symptoms_vec)[0]
            best_match_idx = np.argmax(similarities)
            predicted_disease = cluster_df.iloc[best_match_idx]["disease"]

            st.subheader("Predicted Disease:")
            st.success(predicted_disease)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
################################################################################################################################
if (selected == 'Fetal Health Prediction'):

    # page title
    st.title('Fetal Health Prediction')

    content = "Cardiotocograms (CTGs) are a simple and cost accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality"
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        BaselineValue = st.text_input('Baseline Value')

    with col2:
        Accelerations = st.text_input('Accelerations')

    with col3:
        fetal_movement = st.text_input('Fetal Movement')

    with col1:
        uterine_contractions = st.text_input('Uterine Contractions')

    with col2:
        light_decelerations = st.text_input('Light Decelerations')

    with col3:
        severe_decelerations = st.text_input('Severe Decelerations')

    with col1:
        prolongued_decelerations = st.text_input('Prolongued Decelerations')

    with col2:
        abnormal_short_term_variability = st.text_input('Abnormal Short Term Variability')

    with col3:
        mean_value_of_short_term_variability = st.text_input('Mean Value Of Short Term Variability')

    with col1:
        percentage_of_time_with_abnormal_long_term_variability = st.text_input('Percentage Of Time With ALTV')

    with col2:
        mean_value_of_long_term_variability = st.text_input('Mean Value Long Term Variability')

    with col3:
        histogram_width = st.text_input('Histogram Width')

    with col1:
        histogram_min = st.text_input('Histogram Min')

    with col2:
        histogram_max = st.text_input('Histogram Max')

    with col3:
        histogram_number_of_peaks = st.text_input('Histogram Number Of Peaks')

    with col1:
        histogram_number_of_zeroes = st.text_input('Histogram Number Of Zeroes')

    with col2:
        histogram_mode = st.text_input('Histogram Mode')

    with col3:
        histogram_mean = st.text_input('Histogram Mean')

    with col1:
        histogram_median = st.text_input('Histogram Median')

    with col2:
        histogram_variance = st.text_input('Histogram Variance')

    with col3:
        histogram_tendency = st.text_input('Histogram Tendency')

    # creating a button for Prediction
    st.markdown('</br>', unsafe_allow_html=True)
    with col1:
        if st.button('Predict Pregnancy Risk'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predicted_risk = fetal_model.predict([[BaselineValue, Accelerations, fetal_movement,
                                                       uterine_contractions, light_decelerations, severe_decelerations,
                                                       prolongued_decelerations, abnormal_short_term_variability,
                                                       mean_value_of_short_term_variability,
                                                       percentage_of_time_with_abnormal_long_term_variability,
                                                       mean_value_of_long_term_variability, histogram_width,
                                                       histogram_min, histogram_max, histogram_number_of_peaks,
                                                       histogram_number_of_zeroes, histogram_mode, histogram_mean,
                                                       histogram_median, histogram_variance, histogram_tendency]])
            # st.subheader("Risk Level:")
            st.markdown('</br>', unsafe_allow_html=True)
            if predicted_risk[0] == 0:
                st.markdown(
                    '<bold><p style="font-weight: bold; font-size: 20px; color: green;">Result  Comes to be  Normal</p></bold>',
                    unsafe_allow_html=True)
            elif predicted_risk[0] == 1:
                st.markdown(
                    '<bold><p style="font-weight: bold; font-size: 20px; color: orange;">Result  Comes to be  Suspect</p></Bold>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<bold><p style="font-weight: bold; font-size: 20px; color: red;">Result  Comes to be  Pathological</p><bold>',
                    unsafe_allow_html=True)
    with col2:
        if st.button("Clear"):
            st.rerun()
################################################################################################################################