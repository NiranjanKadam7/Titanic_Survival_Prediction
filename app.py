from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)


lr = pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == "POST":
        Pclass = float(request.form['pclass'])
        Sex = float(request.form['sex'])
        Age = float(request.form['age'])
        SibSp = float(request.form['SibSp'])
        Parch = float(request.form['Parch'])
        Fare = float(request.form['fare'])
        Embarked = float(request.form['embarked'])
        prediction = lr.predict([[Pclass,Sex,Age,SibSp,Parch,Fare,Embarked]])
        pred = prediction[0]
        out = "Error"
        if pred==[1]:out= "Survived"
        else: out = "Not Survived"
        
        return render_template('index.html',results = out)


if __name__ =="__main__":
    app.run(debug=True)
