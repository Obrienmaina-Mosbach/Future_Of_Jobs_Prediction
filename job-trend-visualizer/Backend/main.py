from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained ARIMA model
model = joblib.load("model.pkl")

# Load the job data
job_data = pd.read_csv("postings.csv")

@app.route("/predict", methods=["GET"])
def predict():
    job_title = request.args.get("title", "")
    location = request.args.get("location", "")

    # Filter the data
    filtered_data = job_data[(job_data["title"] == job_title) & (job_data["location"] == location)]

    if filtered_data.empty:
        return jsonify({"error": "No data available for this job title and location"}), 404

    # Use the model to predict future job demand
    future_demand = model.forecast(steps=12)  # Predict next 12 months

    return jsonify({"job_title": job_title, "location": location, "predictions": future_demand.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
