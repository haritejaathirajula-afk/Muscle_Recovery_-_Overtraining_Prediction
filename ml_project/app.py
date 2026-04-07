from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    values = [float(x) for x in request.form.values()]
    final = np.array([values])
    prediction = model.predict(final)

    return render_template('index.html', result=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)