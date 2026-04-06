from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])   # ✅ NO dot here
def predict():
    # Get values from form
    load = float(request.form['load'])
    sleep = float(request.form['sleep'])
    stress = float(request.form['stress'])

    # Prediction
    result = model.predict([[load, sleep, stress]])

    if result[0] == 1:
        output = "✅ Fully Recovered"
    else:
        output = "⚠️ Overtraining Risk"

    return render_template("index.html", prediction=output)

if __name__ == "__main__":
    app.run(debug=True)
