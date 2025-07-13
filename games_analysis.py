import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ttest_ind

# Load dataset
df = pd.read_csv('/mnt/data/games_dataset.csv')

# Data Cleaning & Preprocessing
# Handling missing values
df.dropna(inplace=True)

# Standardizing 'Genre' and 'Platform' columns
df['Genre'] = df['Genre'].str.strip().str.lower()
df['Platform'] = df['Platform'].str.strip().str.lower()

# Creating new feature 'Decade'
df['Release Year'] = df['Release Year'].astype(int)
df['Decade'] = (df['Release Year'] // 10) * 10

# Exploratory Data Analysis
plt.figure(figsize=(12, 6))
sns.countplot(x='Genre', data=df, order=df['Genre'].value_counts().index)
plt.xticks(rotation=90)
plt.title("Game Count by Genre")
plt.show()

# Top 5 Genres & Platforms by Average User Rating
top_genres = df.groupby('Genre')['User Rating'].mean().nlargest(5)
top_platforms = df.groupby('Platform')['User Rating'].mean().nlargest(5)
print("Top 5 Genres by Rating:\n", top_genres)
print("Top 5 Platforms by Rating:\n", top_platforms)

# Advanced Analysis
# Time Series Analysis
plt.figure(figsize=(12, 6))
df.groupby('Decade')['User Rating'].mean().plot(kind='line', marker='o')
plt.title("Average User Rating Over Decades")
plt.xlabel("Decade")
plt.ylabel("Average Rating")
plt.grid()
plt.show()

# Hypothesis Testing
platform_1 = df[df['Platform'] == 'ps4']['User Rating']
platform_2 = df[df['Platform'] == 'xbox one']['User Rating']
t_stat, p_value = ttest_ind(platform_1, platform_2, nan_policy='omit')
print(f'T-test Statistic: {t_stat}, P-value: {p_value}')

# Correlation Analysis
correlation_matrix = df[['Genre', 'Platform', 'Release Year', 'User Rating']].corr()
print("Correlation Matrix:\n", correlation_matrix)

# Machine Learning: Predicting Game Ratings
# Encoding categorical features
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])
df['Platform'] = le.fit_transform(df['Platform'])

# Selecting features & target
X = df[['Genre', 'Platform', 'Release Year']]
y = df['User Rating']

# Splitting data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}, R2 Score: {r2}')
