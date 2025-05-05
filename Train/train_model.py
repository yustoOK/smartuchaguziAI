import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Add, Input
from tensorflow.keras.models import Model
import pickle
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# Preparation of training data
def prepare_data(csv_file):
    df = pd.read_csv(csv_file) #Path to csv file (Containing data)
    
    # Extraction of numeric part of voter_id
    df['voter_id_num'] = df['voter_id'].apply(lambda x: int(x.split('-')[2]))
    
    features = df[['time_diff', 'votes_per_user', 'voter_id_num', 
                  'avg_time_between_votes', 'vote_frequency', 
                  'vpn_usage', 'multiple_logins']].values
    labels = df['label'].values.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, y_train), (X_test_scaled, y_test), scaler

# Data augmentation
def augment_data(X, y, noise_level=0.05):
    X_aug = X.copy()
    noise = np.random.normal(0, noise_level, X.shape)
    X_aug = X_aug + noise
    X_aug = np.clip(X_aug, 0, 1)  # Keep within normalized range
    return X_aug, y

# Residual block
def residual_block(x, units):
    shortcut = x
    x = Dense(units, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(units)(x)
    if shortcut.shape[-1] != units:
        shortcut = Dense(units)(shortcut)
    x = Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    return x

# Creating a model
def create_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = residual_block(x, 64)
    x = residual_block(x, 32)
    
    x = Dense(16, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    
    # Learning rate schedule with warmup
    initial_lr = 0.001
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_lr, decay_steps=10000, alpha=0.01
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)  # Gradient clipping
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# Train model
def train_model(X_train, y_train, X_test, y_test):
    model = create_model(input_shape=(X_train.shape[1],))
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', mode='max', patience=15, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Augment training data
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    X_train_combined = np.vstack([X_train, X_train_aug])
    y_train_combined = np.vstack([y_train, y_train_aug])
    
    history = model.fit(
        X_train_combined, y_train_combined,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=64,  # Larger batch for better gradient estimates
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history
 

#Visualization of Model Architecture 
model = create_model(input_shape=(X_train.shape[1],))
plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)


#Plotting training history
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    # AUC
    plt.subplot(1, 3, 2)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.title('AUC')
    plt.legend()

    # Loss
    plt.subplot(1, 3, 3)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call it after training (Display history)
plot_history(history)



if __name__ == "__main__":
    print("Loading and preparing data...")
    (X_train, y_train), (X_test, y_test), scaler = prepare_data('fraud_data.csv')
    
    print("Training model...")
    model, history = train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Save model and scaler
    model.save('fraud_model.keras')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Model and scaler saved")