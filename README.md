# Future of Jobs Project

![job_trend_visualizer](/job-trend-visualizer/Frontend/assets/future.png)

## Overview
The **Future of Jobs Project** is a web application that predicts job market trends based on historical job postings data. It allows users to filter and analyze job demand by **Job Title, Location, and Skills** using interactive time-series charts. The project integrates **Streamlit for UI, Flask for the backend, and ARIMA for time-series forecasting.**

## Features
- 📊 **Visualize job market trends** using historical data.
- 🔍 **Filter jobs by title, location, and skills.**
- 📈 **Predict future job demand** using an ARIMA model.
- 🔄 **Interactive graphs** for better insights.
- 🛠 **Built using Python, Flask, Streamlit, Pandas, and Plotly.**

## Tech Stack
- **Frontend:** Streamlit (Python)
- **Backend:** Flask
- **Machine Learning:** ARIMA (Time-series forecasting)
- **Data Processing:** Pandas, NumPy, Matplotlib, Seaborn
- **Database:** CSV-based storage (postings.csv)

---

## Project Structure
```
Job-Market-Trend-Visualizer/
│── data/                # Job postings dataset (CSV)
│── Backend/
│   │── main.py          # Flask backend API
│   │── model.pkl        # Trained ARIMA model
    │── postings.csv        # Dataset from Kaggle
    │── requirements.txt     # Dependencies
│── Frontend/
    │── assets/              # Images and UI assets
    │── app.py          # Flask backend API
    │── postings.csv        # Dataset from Kaggle
│   │── model.pkl        # Trained ARIMA model
    │── requirements.txt     # Dependencies
│── README.md            # Project documentation

```

---

## Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Obrienmaina-Mosbach/Future_Of_Jobs_Prediction
cd Future_Of_Jobs_Prediction
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Flask API (Backend)
```sh
cd Backend
python main.py
```

### 4️⃣ Run the Streamlit App (Frontend)
```sh
streamlit run App.py
```

---

## Usage
1. Select a **job title and location** from the dropdown.
2. Click **Predict Future Demand** to generate forecasts.
3. View **interactive charts** displaying job trends.
4. Adjust selections for different job demand insights.

---

## API Endpoint (Flask)
- **Endpoint:** `GET /predict`
- **Parameters:**
  - `title`: Job title (required)
  - `location`: Job location (optional)
- **Example Usage:**
```sh
curl "http://127.0.0.1:5000/predict?title=marketing&location=New%20York"
```
- **Response:** JSON object with predicted job demand.

---

## Machine Learning Model
The **ARIMA (AutoRegressive Integrated Moving Average)** model is trained on historical job postings data to predict future job demand.

- 📌 **Preprocessing:**
  - Converts timestamps into **Year-Month format**.
  - Aggregates job counts by **title, location, and time**.
  - Computes **rolling averages** for trend analysis.

- 📌 **Forecasting:**
  - Uses **ARIMA(1,0,1)** model for time-series prediction.

---

## Contributors
- **Brian Maina Nyawira** *(Project Owner & Developer)*

For suggestions, open an issue or reach out!

📧 Contact: bri.nyawira.24@lehre.mosbach.dhbw.de

---

## License
MIT License. Feel free to use and contribute! 🚀

