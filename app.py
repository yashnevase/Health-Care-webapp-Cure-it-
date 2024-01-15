from flask import Flask, render_template, request, redirect, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from functools import wraps
import os


app = Flask(__name__)

#Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self,email,password,name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))
    
with app.app_context():
    db.create_all()

# Load and preprocess data for Naive Bayes model
df = pd.read_csv("Training.csv")
# Map disease names to numerical labels
disease_mapping = {
    'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
    'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9,
    'Hypertension ': 10, 'Migraine': 11, 'Cervical spondylosis': 12,
    'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17,
    'Typhoid': 18, 'hepatitis A': 19, 'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22,
    'Hepatitis E': 23, 'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
    'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
    'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33,
    'Osteoarthristis': 34, 'Arthritis': 35,
    '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
    'Psoriasis': 39, 'Impetigo': 40
}

# Replace disease names with numerical labels in the dataframe
df.replace({'prognosis': disease_mapping}, inplace=True)

# Features (X) and Target variable (y)
l1 = ['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
    'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue',
    'weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat',
    'irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion',
    'headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation',
    'abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload',
    'swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps',
    'bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain',
    'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side',
    'loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches',
    'watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']  # List of symptoms
X = df[l1]
y = df[["prognosis"]]

# Create and train the Naive Bayes model
gnb = MultinomialNB()
gnb = gnb.fit(X, np.ravel(y))

# Custom decorator to check if the user is logged in
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'email' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

# Creating web pages

# Define the path to the public folder
public_folder_path = os.path.join(os.getcwd(), 'public')

# Serve images from the public folder
@app.route('/public/<path:filename>')
def serve_image(filename):
    return send_from_directory(public_folder_path, filename)

# Landing Page
@app.route('/')
def index():
    return render_template('index.html')

# Registeration Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'email' in session:
        return redirect('/dashboard')
    
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        # Check if the email already exists
        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            return render_template('register.html', error='User already exists. Please log in to continue.')

        # If the email doesn't exist, create a new user
        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        return redirect('/login')

    return render_template('register.html')

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'email' in session:
        return redirect('/dashboard')
    
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user:
            if user.check_password(password):
                session['email'] = user.email
                return redirect('/dashboard')
            else:
                return render_template('login.html', error='Incorrect credentials. Please try again.')
        else:
            return render_template('login.html', error='User does not exist. Please create an account by registering.')

    return render_template('login.html')

#Dashboard
@app.route('/dashboard',methods=['POST','GET'])
@login_required
def dashboard():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        prediction=predict()
        return render_template('dashboard.html',user=user,prediction=prediction, options=l1)
    return redirect('/login')

#Basic Checkup Page
@app.route('/basic_checkup', methods=['POST', 'GET'])
@login_required
def basic_checkup() : 
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        prediction=predict()
        return render_template('basic_checkup.html',user=user,prediction=prediction, options=l1)
    return redirect('/login')

#Predicting disease from symptoms
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    symptoms = [
        request.form.get('symptom1',''), 
        request.form.get('symptom2',''),
        request.form.get('symptom3',''), 
        request.form.get('symptom4',''),
        request.form.get('symptom5','')
    ]

    l2 = [0] * len(l1)

    for k in range(len(l1)):
        for z in symptoms:
            if z == l1[k]:
        
                l2[k] = 1

    input_test = [l2]
  
    print("DEBUG: input_test =", input_test)

    if not input_test or not input_test[0]:
        return render_template('result.html', prediction="Unknown Disease")

    predicted = gnb.predict(input_test)[0]

    disease_prediction = next((key for key, value in disease_mapping.items() if value == predicted), "Unknown Disease")

    return render_template('result.html', prediction=disease_prediction)

#Diabetes Page
@app.route("/diabetes_diagnosis")
@login_required
def diabetes():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        prediction=predict()
        return render_template('diabetes_diagnosis.html',user=user,prediction=prediction, options=l1)
    return redirect('/login')

#Diabetes Result
@app.route('/diabetes_result', methods=['POST'])
def diabetes_result():
    if request.method == 'POST':
         if session['email']:
            user = User.query.filter_by(email=session['email']).first()
            preg = request.form['pregnancies']
            glucose = request.form['glucose']
            bp = request.form['bloodpressure']
            st = request.form['skinthickness']
            insulin = request.form['insulin']
            bmi = request.form['bmi']
            dpf = request.form['dpf']
            age = request.form['age']

            data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
            my_prediction = classifier.predict(data)

            return render_template('diabetes_result.html', prediction=my_prediction, user=user)

df1 = pd.read_csv('diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
df1 = df1.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df1.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure',
                                                                                    'SkinThickness', 'Insulin',
                                                                                    'BMI']].replace(0, np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# Model Building
X = df1.drop(columns='Outcome')
y = df1['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = './static/diabetes_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

#Heart disease predection Page
@app.route('/heart_diagnosis',methods=['GET','POST'])
@login_required
def heart_diagnosis():
    if request.method == 'GET':
        if session['email']:
            user = User.query.filter_by(email=session['email']).first()
            return render_template('heart_diagnosis.html', user=user)
    else:
        if session['email']:
            user = User.query.filter_by(email=session['email']).first()
            age = request.form['age']
            sex = request.form['sex']
            chest = request.form['chest']
            trestbps = request.form['trestbps']
            chol = request.form['chol']
            fbs = request.form['fbs']
            restecg = request.form['restecg']
            thalach = request.form['thalach']
            exang = request.form['exang']
            oldpeak = request.form['oldpeak']
            slope = request.form['slope']
            ca = request.form['ca']
            thal = request.form['thal']
            model2 = pickle.load(open('./static/heart_model.pkl','rb'))
            input_data = [age,sex,chest,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
            for i in range(len(input_data)):
                input_data[i]=float(input_data[i])
            print(input_data)
            input_data_as_numpy_array= np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
            prediction = model2.predict(input_data_reshaped)
            senddata=""
            if (prediction[0]== 0):
                senddata='According to the given details person does not have Heart Disease'
            else:
                senddata='According to the given details chances of having Heart Disease are High, So Please Consult a Doctor'
            return render_template('heart_result.html',resultvalue=senddata, user=user)

#Calculator
@app.route('/calculator')
def calculator():

            
    return render_template('calculator.html')
    
# BMI Calculation & Category identifier

@app.route('/bmi', methods=['POST', 'GET'])
@login_required
def bmi_calculator():
    if request.method == 'POST':
        height = float(request.form['height'])
        weight = float(request.form['weight'])

       
        bmi = round(weight / ((height / 100) ** 2),4)
        if bmi < 18.5:
            category_bmi = "Underweight"
        elif bmi >= 18.5 and bmi < 25:  
            category_bmi = "Healthy"
        elif bmi >= 25 and bmi < 30:
            category_bmi = "Overweight"
        else: 
            category_bmi = "Obese"
    

        return render_template('calculator.html', bmi=bmi,category_bmi=category_bmi)

    return render_template('calculator.html')


#Diabetes Pedigree Function Calculator

@app.route('/calculate_dpf', methods=['POST', 'GET'])
@login_required
def calculate_dpf():
    if request.method == 'POST':
        parent_diabetes = request.form.get('parent_diabetes')
        sibling_diabetes = request.form.get('sibling_diabetes')
        age_onset_relative = int(request.form.get('age_onset_relative'))
        num_relatives_with_diabetes = int(request.form.get('num_relatives_with_diabetes'))

        if parent_diabetes == 'Yes':
            parent_score = 0.2
        else:
            parent_score = 0.0

        if sibling_diabetes == 'Yes': 
            sibling_score = 0.3
        else:
            sibling_score = 0.0
    
        relative_score = 0.5 * (min(age_onset_relative, 40) / 40)

        num_relatives_score = 0.3 * (num_relatives_with_diabetes / 15)

        dpf_score = round(parent_score + sibling_score + relative_score + num_relatives_score, 2)
    
        return render_template('calculator.html', dpf_score=dpf_score)
    
    return render_template('calculator.html')

# Custom 404 error handler
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

#Log out page
@app.route('/logout')
def logout():
    session.pop('email',None)
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)

