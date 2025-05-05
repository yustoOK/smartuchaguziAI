import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pickle

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Select features
    features = [
        'time_diff', 'votes_per_user', 'avg_time_between_votes', 'vote_frequency',
        'vpn_usage', 'multiple_logins', 'session_duration', 'location_flag'
    ]
    X = df[features]
    y = df['label']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Split data: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, features

def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim, kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(X_train, X_val, y_train, y_val):
    class_weights = {0: 1.0, 1: 33.33}  # 1:33 ratio for 3% fraud
    
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
    )
    
    model = build_model(input_dim=X_train.shape[1])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=256,
        class_weight=class_weights,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'], output_dict=True)
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    return y_pred, y_pred_proba, report

def plot_performance(history, y_test, y_pred, y_pred_proba, report):
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_curve.png')
    plt.close()
    
    # Plot precision, recall, F1-score for Fraud class
    fraud_metrics = {
        'Precision': report['Fraud']['precision'],
        'Recall': report['Fraud']['recall'],
        'F1-Score': report['Fraud']['f1-score']
    }
    plt.figure(figsize=(8, 6))
    plt.bar(fraud_metrics.keys(), fraud_metrics.values(), color=['blue', 'green', 'orange'])
    plt.title('Fraud Class Metrics')
    plt.ylabel('Score')
    for i, v in enumerate(fraud_metrics.values()):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.savefig('metrics_bar.png')
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Load and preprocess data
    file_path = 'C:\\Users\\yusto\\Desktop\\fraud_data.csv'
    X_train, X_val, X_test, y_train, y_val, y_test, features = load_and_preprocess_data(file_path)
    
    # Train model
    model, history = train_model(X_train, X_val, y_train, y_val)
    
    # Evaluate and plot
    y_pred, y_pred_proba, report = evaluate_model(model, X_test, y_test)
    plot_performance(history, y_test, y_pred, y_pred_proba, report)
    
    # Save model in .keras format
    model.save('fraud_model.keras')
    
    # Save feature names
    with open('features.pkl', 'wb') as f:
        pickle.dump(features, f)

if __name__ == "__main__":
    main()