import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from wordcloud import WordCloud

# Load job postings dataset
jobs_df = pd.read_csv('./data/postings.csv')

# Convert original_listed_time from milliseconds to datetime format
jobs_df['original_listed_time'] = pd.to_datetime(jobs_df['original_listed_time'], unit='ms')

# Extract year and month
jobs_df['year'] = jobs_df['original_listed_time'].dt.year
jobs_df['month'] = jobs_df['original_listed_time'].dt.month

# Aggregate job postings by job title, job skills and location
job_trends = jobs_df.groupby(['title', 'location', 'year', 'month']).size().reset_index(name='job_count')

# Sort by time
job_trends = job_trends.sort_values(by=['title', 'location', 'year', 'month'])

# Shift job count to use the previous month's demand as a feature
job_trends['job_count_last_month'] = job_trends.groupby(['title', 'location'])['job_count'].shift(1)

# Rolling average of job demand for the past 3 months
job_trends['job_count_avg_3m'] = job_trends.groupby(['title', 'location'])['job_count'].rolling(3).mean().reset_index(drop=True)

# Drop NaN values
#job_trends = job_trends.dropna()

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

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf.predict(X_test)

# Model Evaluation Metrics
mse = mean_squared_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Feature Importance
importances = rf.feature_importances_
feature_names = X_train.columns
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
plt.title('Actual vs Predicted Job Counts')
plt.show()

# Residual Analysis
residuals = y_test - y_pred_rf
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Residual Analysis')
plt.show()

# Predicting future job postings
future_months = 18  # Predict for the next 18 months
last_date = jobs_df['original_listed_time'].max()
future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, future_months + 1)]

# Create a DataFrame for future job postings
future_jobs = []
for date in future_dates:
    for _, row in job_trends.iterrows():
        future_jobs.append({
            'year': date.year,
            'month': date.month,
            'title': row['title'],
            'location': row['location'],
            'job_count_last_month': row['job_count'],  # Last available job count
            'job_count_avg_3m': row['job_count_avg_3m'],  # Rolling average
        })

future_df = pd.DataFrame(future_jobs)

# Predict job counts for the future data
future_df['predicted_job_count'] = rf.predict(future_df[features])

# Display predictions
print(future_df[['year', 'month', 'title', 'location', 'predicted_job_count']])

# Decode job skills to original names
job_trends['title'] = label_encoders['title'].inverse_transform(job_trends['title'])

# Get top 10 skills with highest job postings
top_skills = job_trends.groupby('title')['job_count'].sum().sort_values(ascending=False).head(20)

top_jobs = job_trends.groupby('title')['job_count'].sum().sort_values(ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_jobs.values, y=top_jobs.index, palette='viridis')

plt.xlabel('Total Job Count')
plt.ylabel('Job Title')
plt.title('Top 20 Job Titles in Demand')
plt.grid(axis='x')
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=job_trends, x='month', y='job_count', hue='year', marker='o')

plt.title('Job Demand Trends Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Job Postings')
plt.legend(title='Year')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=future_df, x='month', y='predicted_job_count', hue='year', marker='o')

plt.title('Predicted Job Demand for Next 18 Months')
plt.xlabel('Month')
plt.ylabel('Predicted Job Count')
plt.legend(title='Year')
plt.grid(True)
plt.show()

# Get the top 50 job titles
top_50_titles = job_trends['title'].value_counts().head(50)

# Create a string of job titles for the Word Cloud
job_titles_str = ' '.join(top_50_titles.index)

# Generate the Word Cloud with custom color and font
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(job_titles_str)

# Display the Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top 50 Job Titles Word Cloud')
plt.show()
