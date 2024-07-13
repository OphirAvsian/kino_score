import pandas as pd
import numpy as np

# Define the list of primary metrics and their secondary counterparts
metrics = [
    'bench_press', 'squat', 'deadlift', 'power_clean', '40_yard', '10_yard', 
    'vertical_jump', 'broad_jump', '1_mile', '5k_run', 'bench_press_power', 'squat_power', 
    'deadlift_power', 'power_clean_power', 'body_fat_percentage', 'skeletal_muscle_mass', 
    'quad_asymmetry', 'calf_asymmetry'
]

secondary_metrics = {
    'bench_press': 'incline_bench_press',
    'squat': 'front_squat',
    'deadlift': 'romanian_deadlift',
    'power_clean': 'hang_clean',
    '40_yard': '20_yard_shuttle',
    '10_yard': 'L_drill',
    'vertical_jump': '100_meter',
    'broad_jump': '60_meter',
    '1_mile': 'marathon',
    '5k_run': 'half_marathon',
    'bench_press_power': 'incline_bench_press_power',
    'squat_power': 'front_squat_power',
    'deadlift_power': 'romanian_deadlift_power',
    'power_clean_power': 'hang_clean_power',
    'body_fat_percentage': 'waist_hip_ratio',
    'skeletal_muscle_mass': 'muscle_mass',
    'quad_asymmetry': 'quad_symmetry',
    'calf_asymmetry': 'calf_symmetry'
}

# Define which metrics should be inverted (lower is better)
invert_metrics = [
    '40_yard', '20_yard_shuttle', '10_yard', 'L_drill', 
    '100_meter', '60_meter', '1_mile', '5k_run', 
    'marathon', 'half_marathon', 'body_fat_percentage', 'waist_hip_ratio',
    'quad_asymmetry', 'calf_asymmetry', 'quad_symmetry', 'calf_symmetry',
]

def kino_score(user_data, primary_file='fabricated_v1_data.csv', secondary_file='secondary_metrics.csv'):
    def normalize_and_invert(value, min_value, max_value, invert=False):
        if pd.isna(value) or pd.isna(min_value) or pd.isna(max_value):
            return np.nan
        if invert:
            return (max_value - value) / (max_value - min_value)
        else:
            return (value - min_value) / (max_value - min_value)
    
    # Load primary and secondary metrics data
    primary_df = pd.read_csv(primary_file).drop(columns=['user_id'])
    secondary_df = pd.read_csv(secondary_file).drop(columns=['user_id'])
    
    # all values in user_data are numeric or NaN
    user_data = {k: (float(v) if v != 'n/a' else np.nan) for k, v in user_data.items()}
    
    # Substitute missing primary metrics with secondary metrics in user data
    for primary, secondary in secondary_metrics.items():
        if primary in user_data and pd.isna(user_data[primary]) and secondary in user_data:
            user_data[primary] = user_data[secondary]
    
    # Filter out only the primary metrics
    primary_user_data = {metric: user_data[metric] for metric in metrics if metric in user_data}
    
    # Normalize and invert user data using primary dataset statistics
    normalized_user_data = {}
    for metric in primary_user_data:
        if metric in primary_df.columns:
            min_value = primary_df[metric].min()
            max_value = primary_df[metric].max()
            normalized_user_data[metric] = normalize_and_invert(
                primary_user_data[metric], min_value, max_value, invert=metric in invert_metrics
            )
    
    # Scale normalized values to a 1-100 range
    scaled_user_data = {metric: normalized_user_data[metric] * 100 for metric in normalized_user_data}
    
    # Ensure the values are within the 1-100 range
    scaled_user_data = {metric: np.clip(value, 1, 100) for metric, value in scaled_user_data.items()}
    
    # Calculate KinoScore proportionally based on available metrics
    available_metrics_count = len(scaled_user_data)
    if available_metrics_count == 0:
        return np.nan
    
    kino_score_value = sum(scaled_user_data.values()) / available_metrics_count
    kino_score_value = round(kino_score_value, 2)
    
    return kino_score_value

# Example usage with a dictionary
user_data_dict = {
    'bench_press': 200.0, 'incline_bench_press': 180.0, 'squat': 250.0, 'front_squat': 230.0,
    'deadlift': 300.0, 'romanian_deadlift': 290.0, 'power_clean': 150.0, 'hang_clean': 140.0,
    '40_yard': 4.5, '20_yard_shuttle': 4.2, '10_yard': 1.6, 'L_drill': 7.8,
    'vertical_jump': 30.0, '100_meter': 12.0, 'broad_jump': 110.0, '60_meter': 8.0,
    '1_mile': 7.5, 'marathon': 240.0, '5k_run': 20.0, 'half_marathon': 120.0,
    'bench_press_power': 1000.0, 'incline_bench_press_power': 950.0, 'squat_power': 1500.0, 'front_squat_power': 1400.0,
    'deadlift_power': 1600.0, 'romanian_deadlift_power': 1900.0, 'power_clean_power': 1200.0, 'hang_clean_power': 1300.0,
    'body_fat_percentage': 15.0, 'waist_hip_ratio': 0.9, 'skeletal_muscle_mass': 40.0, 'muscle_mass': 50.0,
    'quad_asymmetry': 5.0, 'quad_symmetry': 95.0, 'calf_asymmetry': 1.30, 'calf_symmetry': 3
}


score = kino_score(user_data_dict)
print(f"KinoScore: {score}")