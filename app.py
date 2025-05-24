from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)

model=pickle.load(open('models/model.pkl','rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])

def predict():
    print("Received POST request")
    mean_radius=float(request.form['mean_radius'])
    mean_texture=float(request.form['mean_texture'])
    mean_perimeter=float(request.form['mean_perimeter'])
    mean_area=float(request.form['mean_area'])
    mean_smoothness=float(request.form['mean_smoothness'])

    features=np.array([[mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness]])

    prediction=model.predict(features)[0]

    diagnosis='malign' if prediction==1 else 'benign'

    return render_template('index.html',prediction_text=f"Breast Cancer Diagnosis: {diagnosis}")

if __name__=="__main__":
    app.run()
