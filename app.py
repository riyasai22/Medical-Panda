import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
modelheart=pickle.load(open('model_heart.pkl','rb'))
modellung=pickle.load(open('model_lung.pkl','rb'))
diabetes=pickle.load(open('diabetes.pkl','rb'))
hepatitis=pickle.load(open('hepatitis.pkl','rb'))
disease=pickle.load(open('disease.pkl','rb'))

@app.route('/',methods=['POST','GET'])
def home():
    if request.method=='POST':
        init_features=[]
        all_features=['receiving_blood_transfusion', 'red_sore_around_nose',
       'abnormal_menstruation', 'continuous_sneezing', 'breathlessness',
       'blackheads', 'shivering', 'dizziness', 'back_pain', 'unsteadiness',
       'yellow_crust_ooze', 'muscle_weakness', 'loss_of_balance', 'chills',
       'ulcers_on_tongue', 'stomach_bleeding', 'lack_of_concentration', 'coma',
       'neck_pain', 'weakness_of_one_body_side', 'diarrhoea',
       'receiving_unsterile_injections', 'headache', 'family_history',
       'fast_heart_rate', 'pain_behind_the_eyes', 'sweating', 'mucoid_sputum',
       'spotting_ urination', 'sunken_eyes', 'dischromic _patches', 'nausea',
       'dehydration', 'loss_of_appetite', 'abdominal_pain', 'stomach_pain',
       'yellowish_skin', 'altered_sensorium', 'chest_pain', 'muscle_wasting',
       'vomiting', 'mild_fever', 'high_fever', 'red_spots_over_body',
       'dark_urine', 'itching', 'yellowing_of_eyes', 'fatigue', 'joint_pain',
       'muscle_pain']
        print(request.form)
        doctor=pd.read_csv('people.csv')
        arr=np.asarray(doctor)
        for y in all_features:
            if y not in request.form.keys():
                init_features.append(0)
            else:
                init_features.append(1)
        print(init_features)
        final_features = [np.array(init_features)]
        prediction = disease.predict(final_features)
        # if (prediction[0] == 0):
        #     output = "Patient doesnt have parkinsons!"
        # else:
        #     output = "Patient have parkinsons!"
        output=prediction[0]
        return render_template('index.html',doctor=arr,len=10,prediction_text='Predicted class:  {val}'.format(val=output))
    else:
        doctor=pd.read_csv('people.csv')
        arr=np.asarray(doctor)
        return render_template('index.html',doctor=arr,len=10)

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        if(prediction[0]==0):
            output="Patient doesnt have parkinsons!"
        else:
            output="Patient have parkinsons!"
        return render_template('parkinson.html', prediction_text=output)
    else:
        return render_template('parkinson.html')
@app.route('/predict2',methods=['POST','GET'])
def predict2():
    if(request.method=='POST'):
        int_features=[float(x) for x in request.form.values()]
        final_feature=[np.array(int_features)]
        prediction=modelheart.predict(final_feature)
        if(prediction[0]==0):
            output="Patient doesnt have heart disease!"
        else:
            output="Patient has heart disease!"
        return render_template('heart.html', prediction_text=output)
    else:
        return render_template('heart.html')

@app.route('/predict3',methods=['POST','GET'])
def predict3():
    if(request.method=='POST'):
        int_features=[float(x) for x in request.form.values()]
        final_feature=[np.array(int_features)]
        prediction=modellung.predict(final_feature)
        if(prediction[0]==0):
            output="Patient doesnt have lung cancer!"
        else:
            output="Patient has lung cancer!"
        return render_template('lung.html', prediction_text=output)
    else:
        return render_template('lung.html')

@app.route('/predict4',methods=['POST','GET'])
def predict4():
    if(request.method=='POST'):
        int_features=[float(x) for x in request.form.values()]
        final_feature=[np.array(int_features)]
        prediction=diabetes.predict(final_feature)
        print(prediction[0])
        if(prediction[0]==0):
            output="Patient doesnt have diabetes!"
        else:
            output="Patient has diabetes!"
        return render_template('diabetes.html', prediction_text=output)
    else:
        return render_template('diabetes.html')

@app.route('/predict5',methods=['POST','GET'])
def predict5():
    if(request.method=='POST'):
        int_features=[float(x) for x in request.form.values()]
        final_feature=[np.array(int_features)]
        prediction=hepatitis.predict(final_feature)
        print(prediction[0])
        if(prediction[0]==0):
            output="Patient doesnt have hepatitis!"
        else:
            output="Patient has disease! {val}".format(val=prediction[0])
        return render_template('hepatitis.html', prediction_text=output)
    else:
        return render_template('hepatitis.html')

if __name__ == "__main__":
    app.run(debug=True)