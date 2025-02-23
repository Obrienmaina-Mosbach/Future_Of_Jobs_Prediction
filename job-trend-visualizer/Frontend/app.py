import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Load job postings data from CSV
postings_df = pd.read_csv("postings.csv")

# Extract unique job titles and locations from the dataframe
job_titles = postings_df['title'].unique()
locations = postings_df['location'].unique()

# Streamlit UI
st.image("./assets/future.png", width=700)  # Adjust the width as needed
st.title("📊 Job Market Trend Visualizer")
st.write("Select a job title and location to view demand trends.")

# User inputs with dropdowns
job_title = st.selectbox("Job Title", job_titles)
location = st.selectbox("Location", locations)

# API request to Flask (modifying the API URL based on location)
if st.button("Predict Future Demand"):
    if location:  # If location is selected, include it in the request
        api_url = f"http://127.0.0.1:5000/predict?title={job_title}&location={location}"
    else:  # If location is not selected, make a request without it
        api_url = f"http://127.0.0.1:5000/predict?title={job_title}"

    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        predictions = pd.DataFrame({"Month": range(1, 13), "Predicted Demand": data["predictions"]})

        # Visualization
        fig = px.line(predictions, x="Month", y="Predicted Demand", title="Predicted Job Demand (Next 12 Months)")
        st.plotly_chart(fig)
    else:
        st.error("No data available for the selected job title and location.")
