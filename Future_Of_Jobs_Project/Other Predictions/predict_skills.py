import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


job_skills = pd.read_csv('../data/job_skills_predict.csv')
print(job_skills.head())

top_skills = job_skills['skill_abr'].value_counts().head(5)

# Extract specific art-related skills
art_skills = job_skills[job_skills['skill_abr'].isin(['DSGN', 'ART'])]
art_skills_count = art_skills['skill_abr'].value_counts()

# Combine top skills and art-related skills
combined_skills = pd.concat([top_skills, art_skills_count])

# Plot the top 10 most required skills
plt.figure(figsize=(12, 6))
sns.barplot(x=combined_skills.index, y=combined_skills.values, palette='viridis')
plt.title('Top 5 Most Required Skills Compared to Art Related Skills')
plt.xlabel('Skill')
plt.ylabel('Number of Jobs')
plt.xticks(rotation=45)
plt.show()