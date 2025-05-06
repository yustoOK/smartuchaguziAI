import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

# === Step 1: Load the dataset ===
df = pd.read_csv('C:\\Users\\yusto\\Desktop\\fraud_data.csv')

# === Step 2: Data preprocessing ===
df = df.drop(columns=['voter_id', 'device_id', 'ip_address', 'vote_timestamp'])  # Drop non-numeric/categorical IDs
df['label'] = df['label'].astype(int)

# Encode any remaining categorical columns
if df['location_flag'].dtype == object:
    df['location_flag'] = LabelEncoder().fit_transform(df['location_flag'])

# === Step 3: Separate features and labels ===
X = df.drop(columns=['label'])
y = df['label']

# === Step 4: Split for model ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 5: Random Forest for feature importance ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# === Feature importance ===
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=True).plot(kind='barh', title='Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()

# === Step 6: PCA for 2D projection ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm', alpha=0.7)
plt.title('PCA Projection of Voting Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Label (0=Normal, 1=Fraud)')
plt.tight_layout()
plt.show()

# === Step 7: Permutation Importance ===
perm_result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

perm_importance = pd.Series(perm_result.importances_mean, index=X.columns)
perm_importance.sort_values(ascending=True).plot(kind='barh', title='Permutation Importance')
plt.tight_layout()
plt.show()
