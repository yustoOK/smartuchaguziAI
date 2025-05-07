import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_training_data(num_samples=100_000, fraud_ratio=0.03):
    data = []
    
    # Simulate election period (48-hour voting window)
    election_start = datetime(2025, 5, 1, 8, 0)  # May 1, 2025, 8 AM
    election_end = election_start + timedelta(hours=48)
    
    # Device/IP pool
    device_pool = [f"device_{i}" for i in range(10000)]
    ip_pool = [f"192.168.{i}.{j}" for i in range(1, 255) for j in range(1, 255)]
    
    for _ in range(num_samples):
        # Generate voter ID
        year = np.random.randint(21, 24)
        number = np.random.randint(0, 100000)
        voter_id = f"T{year}-03-{number:05d}"
        
        # Assign fraud label (3% fraud)
        is_fraud = np.random.random() < fraud_ratio
        
        # Time-based features
        if is_fraud:
            fraud_type = np.random.choice(["bot", "coordinated", "proxy"], p=[0.5, 0.3, 0.2])
            vote_time = election_start + timedelta(seconds=np.random.uniform(0, 48*3600))
            if np.random.random() < 0.6:  # 60% chance of odd-hour fraud
                vote_time = election_start + timedelta(hours=np.random.choice([0, 1, 2, 22, 23]).item())
        else:
            vote_hour = np.random.randint(8, 20)
            vote_time = election_start + timedelta(hours=vote_hour, minutes=np.random.randint(0, 60))
        
        # Behavioral features
        if is_fraud:
            if fraud_type == "bot":
                time_diff = np.random.uniform(0.01, 0.5)  # Very fast
                votes_per_user = np.random.randint(6, 15)  # Exceeds 5
                avg_time_between_votes = np.random.uniform(0.05, 0.3)
                vote_frequency = np.random.uniform(2.0, 5.0)
                session_duration = np.random.uniform(10, 20)
                multiple_logins = np.random.randint(1, 3)
            elif fraud_type == "coordinated":
                time_diff = np.random.uniform(0.5, 2)
                votes_per_user = np.random.randint(6, 10)
                avg_time_between_votes = np.random.uniform(0.3, 1)
                vote_frequency = np.random.uniform(1.0, 2.0)
                session_duration = np.random.uniform(20, 40)
                multiple_logins = np.random.randint(2, 5)
            else:  # Proxy
                time_diff = np.random.uniform(1, 5)
                votes_per_user = np.random.randint(1, 7)
                avg_time_between_votes = np.random.uniform(1, 3)
                vote_frequency = np.random.uniform(0.5, 1.5)
                session_duration = np.random.uniform(30, 60)
                multiple_logins = np.random.randint(1, 4)
            vpn_usage = np.random.choice([0, 1], p=[0.4, 0.6])
            location_flag = np.random.choice([0, 1], p=[0.3, 0.7])
        else:
            time_diff = np.random.uniform(5, 15)  # Time to select candidate
            votes_per_user = np.random.randint(1, 6)  # Up to 5 positions
            avg_time_between_votes = np.random.uniform(5, 10) if votes_per_user > 1 else 0
            vote_frequency = np.random.uniform(0.1, 0.3)
            vpn_usage = np.random.choice([0, 1], p=[0.9, 0.1])
            session_duration = np.random.uniform(60, 180)  # Covers multiple votes
            multiple_logins = np.random.randint(1, 2)  # Typically 1
            location_flag = np.random.choice([0, 1], p=[0.95, 0.05])
        
        # Device/IP features
        device_id = np.random.choice(device_pool)
        ip_address = np.random.choice(ip_pool)
        if is_fraud and fraud_type in ["bot", "coordinated"]:
            device_id = np.random.choice(device_pool[:1000])
            ip_address = np.random.choice(ip_pool[:1000])
        
        # Add noise
        time_diff = max(0.01, time_diff + np.random.normal(0, time_diff * 0.15))
        votes_per_user = max(1, int(votes_per_user + np.random.normal(0, votes_per_user * 0.1)))
        avg_time_between_votes = max(0.01, avg_time_between_votes + np.random.normal(0, avg_time_between_votes * 0.15))
        vote_frequency = max(0.01, vote_frequency + np.random.normal(0, vote_frequency * 0.1))
        session_duration = max(10, session_duration + np.random.normal(0, session_duration * 0.2))
        multiple_logins = max(1, int(multiple_logins + np.random.normal(0, multiple_logins * 0.1)))
        
        # Edge cases
        if np.random.random() < 0.05:
            vpn_usage = 1 - vpn_usage
        if np.random.random() < 0.05:
            location_flag = 1 - location_flag
        
        # Ensure correlation
        vote_frequency = min(vote_frequency, 1.0 / avg_time_between_votes if avg_time_between_votes > 0 else vote_frequency)
        
        data.append([
            time_diff, votes_per_user, voter_id, avg_time_between_votes, vote_frequency,
            vpn_usage, multiple_logins, session_duration, location_flag, device_id, ip_address,
            vote_time.strftime("%Y-%m-%d %H:%M:%S"), is_fraud
        ])
    
    # Create DataFrame
    columns = [
        'time_diff', 'votes_per_user', 'voter_id', 'avg_time_between_votes', 'vote_frequency',
        'vpn_usage', 'multiple_logins', 'session_duration', 'location_flag', 'device_id', 'ip_address',
        'vote_timestamp', 'label'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv('C:\\Users\\yusto\\Desktop\\fraud_data_v5.csv', index=False)
    return df

if __name__ == "__main__":
    print("Generating data...")
    df = generate_training_data()
    print("Data saved to fraud_data_v5.csv")
    print(f"Fraud ratio: {df['label'].mean():.3f}")


    #Checking the summary of the data
    