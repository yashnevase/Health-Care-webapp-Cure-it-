# Health-Care-webapp-Cure-it-
Cure-It is a disease prediction web application that allows users to input their symptoms and health parameters to get assessed for potential medical conditions. The app provides risk analysis for 3 major disease areas - general illness based on symptoms, heart disease, and diabetes.

# Cure-It
Cure-It is a web application built with Flask that allows users to get predictions on various diseases based on their symptoms and health data.

# Overview
The app provides disease predictions in 3 main areas:

General Disease Prediction
Heart Disease Prediction
Diabetes Prediction
Users can create an account and then input their symptoms or details like blood pressure, glucose levels etc to get a disease risk assessment. The predictions are made using Machine Learning models like Naive Bayes, Random Forest and Logistic Regression.

# Installation
Clone the repository and install dependencies:
git clone https://github.com/<your_repo>
cd cure-it
pip install -r requirements.txt

# Usage

Run the app:

python app.py

The app will be served at http://localhost:5000

Register an account and then navigate to the disease prediction pages. Enter symptoms and details as prompted to get the disease risk prediction.


# Models

The following ML models are used for the app:

Naive Bayes - Disease prediction based on symptoms
Random Forest - Diabetes risk prediction
Logistic Regression - Heart disease risk prediction
The models are trained onstandard datasets and saved as Pickle files.

# Technology Stack

Frontend: HTML
Backend: Flask, Python
Database: SQLite
Machine Learning: scikit-learn, Pandas, NumPy
