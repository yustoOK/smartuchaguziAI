import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("C:\\Users\\yusto\\Desktop\\fraud_data.csv")

# Convert vote_timestamp to datetime
df['vote_timestamp'] = pd.to_datetime(df['vote_timestamp'])

# Convert 'label' column to string for plotting
df['label'] = df['label'].astype(str)

# 1. Fraud vs Normal count
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.title("Fraud vs Normal Counts")
plt.xlabel("Is Fraud?")
plt.ylabel("Number of Records")
plt.show()

# 2. Time-based distribution of votes
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='vote_timestamp', hue='label', bins=30, kde=True)
plt.title("Vote Timestamps by Label")
plt.xlabel("Timestamp")
plt.ylabel("Vote Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Distribution of time_diff
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x='label', y='time_diff')
plt.title("Time Difference vs Fraud Label")
plt.xlabel("Is Fraud?")
plt.ylabel("Time Difference")
plt.show()

# 4. Correlation heatmap (numeric features)
plt.figure(figsize=(12, 6))
numerics = df.select_dtypes(include='number')
correlation = numerics.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
