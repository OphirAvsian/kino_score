import pandas as pd
import numpy as np

np.random.seed(42)

# Define the number of users
num_users = 100

# Fabricate data for each metric
data = {
    'user_id': range(1, num_users + 1),
    'bench_press': np.random.randint(100, 300, num_users),
    'squat': np.random.randint(150, 350, num_users),
    'deadlift': np.random.randint(200, 400, num_users),
    'power_clean': np.random.randint(120, 250, num_users),
    '40_yard': np.random.uniform(4.0, 6.0, num_users).round(2),
    '10_yard': np.random.uniform(1.5, 2.5, num_users).round(2),
    'vertical_jump': np.random.uniform(20, 40, num_users).round(2),
    'broad_jump': np.random.uniform(60, 100, num_users).round(2),
    '1_mile': np.random.uniform(5.5, 9.0, num_users).round(2),
    '5k_run': np.random.uniform(20.0, 35.0, num_users).round(2),
    'bench_press_power': np.random.uniform(300, 600, num_users).round(2),
    'squat_power': np.random.uniform(500, 900, num_users).round(2),
    'deadlift_power': np.random.uniform(600, 1000, num_users).round(2),
    'power_clean_power': np.random.uniform(400, 700, num_users).round(2),
    'body_fat_percentage': np.random.uniform(15.0, 25.0, num_users).round(2),
    'skeletal_muscle_mass': np.random.randint(120, 200, num_users),
    'quad_asymmetry': np.random.uniform(0.5, 2.0, num_users).round(2),
    'calf_asymmetry': np.random.uniform(0.3, 1.5, num_users).round(2),
    'sleep_quality': np.random.uniform(1, 10, num_users).round(2),
    'energy_levels': np.random.uniform(1, 10, num_users).round(2),
    'stress_levels': np.random.uniform(1, 10, num_users).round(2),
    'mood': np.random.uniform(1, 10, num_users).round(2),
    'focus': np.random.uniform(1, 10, num_users).round(2),
    'pain_levels': np.random.uniform(1, 10, num_users).round(2),
    'consistency': np.random.uniform(1, 10, num_users).round(2),
    'steps': np.random.randint(5000, 15000, num_users),
    'walking_distance': np.random.uniform(3, 10, num_users).round(2),
    'flights_climbed': np.random.randint(5, 50, num_users),
    'max_heart_rate': np.random.randint(160, 200, num_users),
    'resting_heart_rate': np.random.randint(50, 70, num_users),
    'heart_rate_variability': np.random.uniform(20, 100, num_users).round(2),
    'calories': np.random.randint(1500, 3500, num_users),
    'protein': np.random.randint(50, 200, num_users),
    'carbohydrates': np.random.randint(200, 400, num_users),
    'fats': np.random.randint(50, 100, num_users),
    'hydration': np.random.uniform(1, 5, num_users).round(2)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV file
df.to_csv('fabricated_v1_data.csv', index=False)

print(df.head())

# Number of fabricated users
num_users = 100

# Generate realistic ranges for secondary metrics
data_sec = {
    'user_id': np.arange(1, num_users + 1),
    'incline_bench_press': np.round(np.random.uniform(100, 300, num_users), 2),
    'front_squat': np.round(np.random.uniform(150, 350, num_users), 2),
    'romanian_deadlift': np.round(np.random.uniform(200, 400, num_users), 2),
    'hang_clean': np.round(np.random.uniform(100, 250, num_users), 2),
    '20_yard_shuttle': np.round(np.random.uniform(3.5, 6.0, num_users), 2),
    'L_drill': np.round(np.random.uniform(6.5, 9.0, num_users), 2),
    '100_meter': np.round(np.random.uniform(10, 15, num_users), 2),
    '60_meter': np.round(np.random.uniform(7, 10, num_users), 2),
    'marathon': np.round(np.random.uniform(150, 300, num_users), 2),
    'half_marathon': np.round(np.random.uniform(60, 150, num_users), 2),
    'incline_bench_press_power': np.round(np.random.uniform(200, 400, num_users), 2),
    'front_squat_power': np.round(np.random.uniform(400, 700, num_users), 2),
    'romanian_deadlift_power': np.round(np.random.uniform(500, 800, num_users), 2),
    'hang_clean_power': np.round(np.random.uniform(300, 600, num_users), 2),
    'waist_hip_ratio': np.round(np.random.uniform(0.8, 1.2, num_users), 2),
    'muscle_mass': np.round(np.random.uniform(50, 200, num_users), 2),
    'quad_symmetry': np.round(np.random.uniform(0.5, 2.0, num_users), 2),
    'calf_symmetry': np.round(np.random.uniform(0.5, 2.0, num_users), 2)
}

# Create a DataFrame
secondary_df = pd.DataFrame(data_sec)

# Save to a CSV file
secondary_df.to_csv('secondary_metrics.csv', index=False)
print("Secondary metrics data saved to 'secondary_metrics.csv'")
