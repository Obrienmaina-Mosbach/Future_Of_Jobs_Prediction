import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load dataset
jobs_df = pd.read_csv('./data/postings.csv')

# Convert time column
jobs_df['original_listed_time'] = pd.to_datetime(jobs_df['original_listed_time'], unit='ms')

# Extract year and month
jobs_df['year'] = jobs_df['original_listed_time'].dt.year
jobs_df['month'] = jobs_df['original_listed_time'].dt.month

# Aggregate job postings
job_trends = jobs_df.groupby(['title', 'location', 'year', 'month']).size().reset_index(name='job_count')
job_trends = job_trends.sort_values(by=['title', 'location', 'year', 'month'])

# Add features
job_trends['job_count_last_month'] = job_trends.groupby(['title', 'location'])['job_count'].shift(1).fillna(0)
job_trends['job_count_avg_3m'] = job_trends.groupby(['title', 'location'])['job_count'].rolling(3).mean().reset_index(drop=True).fillna(0)

# Encode categorical features
label_encoders = {}
for col in ['title', 'location']:
    le = LabelEncoder()
    job_trends[col] = le.fit_transform(job_trends[col])
    label_encoders[col] = le

# Select features & target
features = ['year', 'month', 'title', 'location', 'job_count_last_month', 'job_count_avg_3m']
X = job_trends[features]
y = job_trends['job_count']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb.predict(X_test)

# Evaluation
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f'XGBoost MSE: {mse_xgb}, R^2: {r2_xgb}')

# Feature Importance
importances = xgb.feature_importances_
feature_names = features
sorted_indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 5))
sns.barplot(x=importances[sorted_indices], y=np.array(feature_names)[sorted_indices], palette='Blues_r')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# Job postings over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=job_trends, x='month', y='job_count', hue='year', marker='o')
plt.title('Monthly Job Postings Trend')
plt.xlabel('Month')
plt.ylabel('Job Count')
plt.legend(title='Year')
plt.show()

# Job count distribution
plt.figure(figsize=(8, 5))
sns.histplot(job_trends['job_count'], bins=30, kde=True, color='blue')
plt.title('Distribution of Job Counts')
plt.xlabel('Job Count')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(job_trends.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# Residual plot
residuals = y_test - y_pred_xgb
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred_xgb, y=residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Job Count')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Prediction error histogram
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color='red')
plt.title('Prediction Error Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.show()
