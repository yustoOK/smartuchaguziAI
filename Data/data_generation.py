import numpy as np
import pandas as pd

def generate_training_data(num_samples=1500000):
    np.random.seed(42)
    
    data = []
    
    for _ in range(num_samples):
        year = np.random.randint(21, 24)
        number = np.random.randint(0, 100000)
        voter_id = f"T{year}-03-{number:05d}"
        
        # Base behavior
        is_fraud = np.random.random() > 0.5
        if not is_fraud:  # Normal
            time_diff = np.random.uniform(5, 30)
            votes_per_user = np.random.randint(1, 3)
            avg_time_between_votes = np.random.uniform(4, 20)
            vote_frequency = np.random.uniform(0.1, 0.5)
            vpn_usage = np.random.choice([0, 1], p=[0.8, 0.2])  # 20% chance of VPN
            multiple_logins = np.random.randint(1, 3)  # 1-2 logins
            label = 0
        else:  # Fraud
            time_diff = np.random.uniform(0.1, 3)
            votes_per_user = np.random.randint(3, 10)
            avg_time_between_votes = np.random.uniform(0.1, 2)
            vote_frequency = np.random.uniform(0.8, 2.0)
            vpn_usage = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% chance of VPN
            multiple_logins = np.random.randint(2, 6)  # 2-5 logins
            label = 1
        
        # Add noise
        noise_factor = 0.1  # 10% noise
        time_diff = max(0.1, time_diff + np.random.normal(0, time_diff * noise_factor))
        votes_per_user = max(1, int(votes_per_user + np.random.normal(0, votes_per_user * noise_factor)))
        avg_time_between_votes = max(0.1, avg_time_between_votes + np.random.normal(0, avg_time_between_votes * noise_factor))
        vote_frequency = max(0.01, vote_frequency + np.random.normal(0, vote_frequency * noise_factor))
        multiple_logins = max(1, int(multiple_logins + np.random.normal(0, multiple_logins * noise_factor)))
        
        # Random flips for VPN (simulating edge cases)
        if np.random.random() < 0.05:  # 5% chance to flip VPN usage
            vpn_usage = 1 - vpn_usage
        
        data.append([time_diff, votes_per_user, voter_id, avg_time_between_votes, 
                    vote_frequency, vpn_usage, multiple_logins, label])
    
    df = pd.DataFrame(data, columns=['time_diff', 'votes_per_user', 'voter_id',
                                   'avg_time_between_votes', 'vote_frequency', 
                                   'vpn_usage', 'multiple_logins', 'label'])
    df.to_csv('fraud_data.csv', index=False)
    return df

if __name__ == "__main__":
    print("Generating data...")
    generate_training_data()
    print("Data saved to fraud_data.csv")