from flask import Flask,request,jsonify
import numpy as np
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model("catboost_model", format='cbm')
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    rri = request.form.get('rri')
    rri = rri[1:-1].split(",")

    result = list()
    for y in rri:
        result.append(float(y))

    result = model.predict(result)
    return jsonify({'placement':str(result[0])})
if __name__ == '__main__':
    app.run(debug=True)