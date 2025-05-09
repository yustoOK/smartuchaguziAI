import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_training_data(num_samples=100_000, fraud_ratio=0.03):
    data = []
    
    # Simulate election period (48-hour voting window)
    election_start = datetime(2025, 6, 25, 8, 0)  # May 1, 2025, 8 AM
    election_end = election_start + timedelta(hours=48)
    
    for _ in range(num_samples):
        # Assign fraud label (3% fraud)
        is_fraud = np.random.random() < fraud_ratio
        
        # Time-based features (vote_timestamp not stored)
        if is_fraud:
            fraud_type = np.random.choice(["bot", "coordinated", "proxy"], p=[0.5, 0.3, 0.2])
            if np.random.random() < 0.6:  # 60% chance of odd-hour fraud
                vote_time = election_start + timedelta(hours=np.random.choice([0, 1, 2, 22, 23]).item())
            else:
                vote_time = election_start + timedelta(seconds=np.random.uniform(0, 48*3600))
        else:
            vote_hour = np.random.randint(8, 20)
            vote_time = election_start + timedelta(hours=vote_hour, minutes=np.random.randint(0, 60))
        
        # Behavioral features
        if is_fraud:
            if fraud_type == "bot":
                time_diff = np.random.uniform(0.01, 0.5)
                votes_per_user = np.random.randint(6, 15)
                vote_frequency = np.random.uniform(2.0, 5.0)
                session_duration = np.random.uniform(10, 20)
                multiple_logins = np.random.randint(1, 3)
            elif fraud_type == "coordinated":
                time_diff = np.random.uniform(0.5, 2)
                votes_per_user = np.random.randint(6, 10)
                vote_frequency = np.random.uniform(1.0, 2.0)
                session_duration = np.random.uniform(20, 40)
                multiple_logins = np.random.randint(2, 5)
            else:  # Proxy
                time_diff = np.random.uniform(1, 5)
                votes_per_user = np.random.randint(1, 7)
                vote_frequency = np.random.uniform(0.5, 1.5)
                session_duration = np.random.uniform(30, 60)
                multiple_logins = np.random.randint(1, 4)
            vpn_usage = np.random.choice([0, 1], p=[0.4, 0.6])
            location_flag = np.random.choice([0, 1], p=[0.3, 0.7])
        else:
            time_diff = np.random.uniform(5, 15)
            votes_per_user = np.random.randint(1, 6)
            vote_frequency = np.random.uniform(0.1, 0.3)
            vpn_usage = np.random.choice([0, 1], p=[0.9, 0.1])
            session_duration = np.random.uniform(60, 180)
            multiple_logins = np.random.randint(1, 2)
            location_flag = np.random.choice([0, 1], p=[0.95, 0.05])
        
        # Add noise
        time_diff = max(0.01, time_diff + np.random.normal(0, time_diff * 0.15))
        votes_per_user = max(1, int(votes_per_user + np.random.normal(0, votes_per_user * 0.1)))
        vote_frequency = max(0.01, vote_frequency + np.random.normal(0, vote_frequency * 0.1))
        session_duration = max(10, session_duration + np.random.normal(0, session_duration * 0.2))
        multiple_logins = max(1, int(multiple_logins + np.random.normal(0, multiple_logins * 0.1)))
        
        # Edge cases
        if np.random.random() < 0.05:
            vpn_usage = 1 - vpn_usage
        if np.random.random() < 0.05:
            location_flag = 1 - location_flag
        
        data.append([
            time_diff, votes_per_user, vote_frequency, vpn_usage,
            multiple_logins, session_duration, location_flag, is_fraud
        ])
    
    # Create DataFrame
    columns = [
        'time_diff', 'votes_per_user', 'vote_frequency', 'vpn_usage',
        'multiple_logins', 'session_duration', 'location_flag', 'label'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv('C:\\Users\\yusto\\Desktop\\fraud_data.csv', index=False)
    return df

if __name__ == "__main__":
    print("Generating data...")
    df = generate_training_data()
    print("Data saved to fraud_data.csv")
    print(f"Fraud ratio: {df['label'].mean():.3f}")