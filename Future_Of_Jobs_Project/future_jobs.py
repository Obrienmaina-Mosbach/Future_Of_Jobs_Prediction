import numpy as np
import pandas as pd
import datetime

from matplotlib import lines
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from webencodings import labels
import matplotlib.dates as mdates
import seaborn as sns

# Load job postings dataset
jobs_df = pd.read_csv("./data/job_descriptions.csv")

# Convert date_posted to datetime format
jobs_df["Job Posting Date"] = pd.to_datetime(jobs_df["Job Posting Date"], errors="coerce")

# Extract year and month
jobs_df["year"] = jobs_df["Job Posting Date"].dt.year
jobs_df["month"] = jobs_df["Job Posting Date"].dt.month

# Aggregate job postings by job title, job skills and location
job_trends = jobs_df.groupby(["skills", "Job Title", "location", "year", "month"]).size().reset_index(name="job_count")

# Sort by time
job_trends = job_trends.sort_values(by=["skills", "Job Title", "location", "year", "month"])

# Shift job count to use the previous month's demand as a feature
job_trends["job_count_last_month"] = job_trends.groupby(["skills", "Job Title", "location"])["job_count"].shift(1)

# Rolling average of job demand for the past 3 months
job_trends["job_count_avg_3m"] = job_trends.groupby(["skills", "Job Title", "location"])["job_count"].rolling(3).mean().reset_index(drop=True)

# Encode categorical variables
label_encoders = {}
for col in ["skills", "Job Title", "location", "year", "month"]:
    le = LabelEncoder()
    le.fit(jobs_df[col].unique())  # Fit with all unique values
    job_trends[col] = le.transform(job_trends[col])
    label_encoders[col] = le  # Store encoders for later use

# Drop rows with NaN values
job_trends.dropna(subset=["job_count_last_month", "job_count_avg_3m"], inplace=True)

# Select features & target
features = ["year", "month", "skills", "Job Title", "location", "job_count_last_month", "job_count_avg_3m"]
X = job_trends[features]
y = job_trends["job_count"]

# Split data into training & testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

# Predicting future job postings
future_months = 24  # Predict for the next 6 months
last_date = jobs_df["Job Posting Date"].max()
future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, future_months + 1)]

# Create a DataFrame for future job postings
future_jobs = []
for date in future_dates:
    for _, row in job_trends.iterrows():
        future_jobs.append({
            "year": date.year,
            "month": date.month,
            "skills": row["skills"],
            "Job Title": row["Job Title"],
            "location": row["location"],
            "job_count_last_month": row["job_count"],  # Last available job count
            "job_count_avg_3m": row["job_count_avg_3m"],  # Rolling average
        })

future_df = pd.DataFrame(future_jobs)

# Predict job counts for the future data
future_df["predicted_job_count"] = rf.predict(future_df[features])

# Display predictions
print(future_df[["year", "month", "skills", "Job Title", "location", "predicted_job_count"]])

# Decode job skills to original names
job_trends["Job Title"] = label_encoders["Job Title"].inverse_transform(job_trends["Job Title"])

# Get top 10 skills with highest job postings
top_skills = job_trends.groupby("Job Title")["job_count"].sum().sort_values(ascending=False).head(20)

top_jobs = job_trends.groupby("Job Title")["job_count"].sum().sort_values(ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_jobs.values, y=top_jobs.index, palette="viridis")

plt.xlabel("Total Job Count")
plt.ylabel("Job Title")
plt.title("Top 20 Job Titles in Demand")
plt.grid(axis="x")
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=job_trends, x="month", y="job_count", hue="year", marker="o")

plt.title("Job Demand Trends Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Job Postings")
plt.legend(title="Year")
plt.grid(True)
plt.show()

importances = rf.feature_importances_
feature_names = features

plt.figure(figsize=(10, 5))
sns.barplot(x=importances, y=feature_names, palette="Blues_r")

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Job Demand Prediction")
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=future_df, x="month", y="predicted_job_count", hue="year", marker="o")

plt.title("Predicted Job Demand for Next 24 Months")
plt.xlabel("Month")
plt.ylabel("Predicted Job Count")
plt.legend(title="Year")
plt.grid(True)
plt.show()