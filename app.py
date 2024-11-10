import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import sklearn

app = Flask(__name__)
data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bathrooms'])
    sqft = float(request.form['total_sqft'])
    print(location,bhk,bath,sqft)
    input = pd.DataFrame([[location,bhk,bath,sqft]],columns=['location','bhk','bath','total_sqft'])
    prediction = pipe.predict(input)[0]*1e5

    return str(np.round(prediction,2))
if __name__ == "__main__":
    app.run(debug=True, port=5000)