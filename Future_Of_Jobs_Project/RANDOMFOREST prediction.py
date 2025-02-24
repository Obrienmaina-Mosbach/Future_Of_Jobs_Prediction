import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from wordcloud import WordCloud
from scipy.stats import randint

# Load job postings dataset
jobs_df = pd.read_csv('./data/postings.csv')

# Convert original_listed_time from milliseconds to datetime format
jobs_df['original_listed_time'] = pd.to_datetime(jobs_df['original_listed_time'], unit='ms')

# Extract year and month
jobs_df['year'] = jobs_df['original_listed_time'].dt.year
jobs_df['month'] = jobs_df['original_listed_time'].dt.month

# Aggregate job postings by job title, job skills, and location
job_trends = jobs_df.groupby(['title', 'location', 'year', 'month']).size().reset_index(name='job_count')

# Sort by time
job_trends = job_trends.sort_values(by=['title', 'location', 'year', 'month'])

# Shift job count to use the previous month's demand as a feature
job_trends['job_count_last_month'] = job_trends.groupby(['title', 'location'])['job_count'].shift(1)

# Rolling average of job demand for the past 3 months
job_trends['job_count_avg_3m'] = job_trends.groupby(['title', 'location'])['job_count'].rolling(3).mean().reset_index(drop=True)

# Handle missing values by making them 0
job_trends['job_count_last_month'] = job_trends['job_count_last_month'].fillna(0)
job_trends['job_count_avg_3m'] = job_trends['job_count_avg_3m'].fillna(0)

# Ensure no all-NaN slices
job_trends = job_trends.dropna(subset=['job_count_last_month', 'job_count_avg_3m'])

# Encode categorical variables
label_encoders = {}
for col in ['title', 'location']:
    le = LabelEncoder()
    job_trends[col] = le.fit_transform(job_trends[col])
    label_encoders[col] = le

# Select features & target
features = ['year', 'month', 'title', 'location', 'job_count_last_month', 'job_count_avg_3m']
X = job_trends[features]
y = job_trends['job_count']

# Feature Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest
param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10)
}

rf = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=3, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)
rf_best = random_search.best_estimator_

# Train ExtraTreesRegressor for comparison
et = ExtraTreesRegressor(n_estimators=200, random_state=42)
et.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_best.predict(X_test)
y_pred_et = et.predict(X_test)

# Model Evaluation Metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Random Forest MSE: {mse_rf}, R^2: {r2_rf}')

mse_et = mean_squared_error(y_test, y_pred_et)
r2_et = r2_score(y_test, y_pred_et)
print(f'ExtraTreesRegressor MSE: {mse_et}, R^2: {r2_et}')

# Feature Importance
importances = rf_best.feature_importances_
feature_names = features
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
sns.barplot(x=importances[sorted_indices], y=np.array(feature_names)[sorted_indices], palette='Blues_r')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Feature Importance in Job Demand Prediction')
plt.show()

# Actual vs Predicted Job Counts
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Job Count')
plt.ylabel('Predicted Job Count')
plt.title('Actual vs Predicted Job Counts (Random Forest)')
plt.show()

# Word Cloud for Job Titles
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(jobs_df['title']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Job Titles')
plt.show()

# Time Series Plot for Job Demand
plt.figure(figsize=(12, 6))
sns.lineplot(data=job_trends, x='month', y='job_count', hue='year', palette='coolwarm')
plt.xlabel('Month')
plt.ylabel('Job Count')
plt.title('Job Demand Trends Over Time')
plt.legend(title='Year')
plt.show()

# Distribution of Job Counts
plt.figure(figsize=(8, 5))
sns.histplot(job_trends['job_count'], bins=30, kde=True, color='blue')
plt.xlabel('Job Count')
plt.ylabel('Frequency')
plt.title('Distribution of Job Postings')
plt.show()

# Residual Plot for Model Errors
residuals_rf = y_test - y_pred_rf
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=residuals_rf, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Actual Job Count')
plt.ylabel('Residuals')
plt.title('Residual Plot (Random Forest)')
plt.show()

# Error Distribution
plt.figure(figsize=(8, 5))
sns.histplot(residuals_rf, bins=30, kde=True, color='red')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution (Random Forest)')
plt.show()