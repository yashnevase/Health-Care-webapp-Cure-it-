# Health-Care-webapp-Cure-it-
Cure-It is a disease prediction web application that allows users to input their symptoms and health parameters to get assessed for potential medical conditions. The app provides risk analysis for 3 major disease areas - general illness based on symptoms, heart disease, and diabetes.

# Cure-It
Cure-It is a web application built with Flask that allows users to get predictions on various diseases based on their symptoms and health data.

# Overview
Images:
![index](https://github.com/yashnevase/Health-Care-webapp-Cure-it-/assets/78201930/84187523-c7f5-4d46-9cc6-bb5fad792efe)
![reg](https://github.com/yashnevase/Health-Care-webapp-Cure-it-/assets/78201930/16b0fcd9-985e-42d6-9703-b6250b307ed0)
![dash](https://github.com/yashnevase/Health-Care-webapp-Cure-it-/assets/78201930/a2284900-9c40-4f04-ad73-1f1fdf911ae3)
![bassic](https://github.com/yashnevase/Health-Care-webapp-Cure-it-/assets/78201930/09eb114d-e256-451c-bd83-2e4b0f779132)
![heart](https://github.com/yashnevase/Health-Care-webapp-Cure-it-/assets/78201930/05b7743f-c8f2-4261-b600-60569c7c3b02)
![diabetes](https://github.com/yashnevase/Health-Care-webapp-Cure-it-/assets/78201930/d46401c1-70b6-4f80-bd02-258cf213dd67)
![cal](https://github.com/yashnevase/Health-Care-webapp-Cure-it-/assets/78201930/c8753029-07ae-480e-b8b8-3900cd10d0af)


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
