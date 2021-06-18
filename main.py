from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('a1.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('r2.html')
standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Gender=request.form['Gender']
        if(Gender=='Male'):
            Gender=1
        else:
            Gender=0
        Age = int(request.form['Age'])
        openness=int(request.form['openness'])
        neuroticism=int(request.form['neuroticism'])
        conscientiousness=int(request.form['conscientiousness'])
        agreeableness=int(request.form['agreeableness'])
        extraversion=int(request.form['extraversion'])

    prediction = model.predict([[Gender,Age,openness,neuroticism,conscientiousness,agreeableness,extraversion]])
    output = prediction[0]
    return render_template('r2.html',prediction_text="Your are a {} person".format(output))

   




if __name__=="__main__":
    app.run(debug=True)