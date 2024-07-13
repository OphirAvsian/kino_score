import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

# def get_user_inputs():
#     
#     user_data = {}
#     print("Please enter the following metrics for the user and if you don't have them type: 'n/a':")
#     for metric in metrics:
#         value = input(f"{metric}:")
#         if value.lower() == 'n/a' and metric in secondary_metrics:
#             value = input(f"{secondary_metrics[metric]}:")
#             user_data[secondary_metrics[metric]] = value
#         else:
#             user_data[metric] = value
            
#     return user_data

# def save_user_data(user_data, filename='user_data.csv'):
#     user_df = pd.DataFrame(user_data, index=[0])
#     user_df.to_csv(filename, index=False)
#     print("User data saved successfully!")

def load_and_process_data(user_df):
    # Load primary and secondary metrics data
    primary_df = pd.read_csv("fabricated_v1_data.csv").drop(columns=['user_id'])
    secondary_df = pd.read_csv('secondary_metrics.csv').drop(columns=['user_id'])
    # user_df = pd.read_csv(user_file)
    
    
    # Combine primary and user data
    combined_df = pd.concat([primary_df, user_df], ignore_index=True)
    
    # Substitute missing primary metrics with secondary metrics
    for primary, secondary in secondary_metrics.items():
        if primary in combined_df.columns and secondary in user_df.columns:
            combined_df[primary] = combined_df[primary].fillna(combined_df[secondary])
    
    present_metrics = [metric for metric in metrics if metric in combined_df.columns and combined_df[metric].notnull().any()]
    
    # Define which metrics should be inverted (lower is better)
    invert_metrics = [
        '40_yard', '20_yard_shuttle',
        '10_yard', 'L_drill',
        '1_mile', 'marathon',
        '5k_run', 'half_marathon',
        'body_fat_percentage', 'waist_hip_ratio'
    ]

    # Normalize and invert metrics as needed
    def normalize_and_invert(x, invert=False):
        if invert: return (x.max() - x) / (x.max() - x.min())
        else: return (x - x.min()) / (x.max() - x.min())

    normalized_metrics = combined_df[present_metrics].apply(lambda x: normalize_and_invert(x, invert=x.name in invert_metrics))
    
    # Scale normalized values to a 1-100 range
    scaled_metrics = normalized_metrics * 100
    
    # Ensure the values are within the 1-100 range
    scaled_metrics = scaled_metrics.apply(lambda x: np.clip(x, 1, 100))
    
    # Calculate KinoScore proportionally based on available metrics
    kino_score = scaled_metrics.apply(lambda row: row.sum() / len(row.dropna()), axis=1).round(2)
    combined_df['KinoScore'] = kino_score
    
    combined_df.to_csv('combined_data_w_score.csv', index=False)
    print(f"Updated metrics with KinoScore saved to 'combined_data_w_score.csv'")
    print(combined_df[['KinoScore']].iloc[-1])
    
def main():
    # user_data = get_user_inputs()
    user_data = [100,150,200,100,None,2.5,20.0,60.0,8.0,30.0,1.5,2.0,2.5,2.0,25.0,120,2.0,None]
    # save_user_data(user_data)
    user_df = pd.DataFrame([user_data], columns=metrics)
    load_and_process_data(user_df)

main()
"""
def test_kino_score():
    test_cases = [
        ('Professional Athlete', {
            'bench_press': 300,
            'squat': 350,
            'deadlift': 450,
            'power_clean': 250,
            '40_yard': 4.0,
            '10_yard': 1.5,
            'vertical_jump': 40,
            'broad_jump': 100,
            '1_mile': 5,
            '5k_run': 18,
            'bench_press_power': 3.5,
            'squat_power': 4.0,
            'deadlift_power': 4.5,
            'power_clean_power': 3.5,
            'body_fat_percentage': 14.0,
            'skeletal_muscle_mass': 200,
            'quad_asymmetry': 0.3,
            'calf_asymmetry': 0.3,
            'hang_clean': 250,
            'L_drill': 6.0,
            'incline_bench_press_power': 3.0,
            'front_squat_power': 4.0,
            'romanian_deadlift_power': 4.5,
            'hang_clean_power': 3.5,
            'waist_hip_ratio': 0.85,
            'muscle_mass': 190,
            'quad_symmetry': 0.3,
            'calf_symmetry': 0.3
        }, (90, 100)),
        ('Highly Fit Individual', {
            'bench_press': 250,
            'squat': 300,
            'deadlift': 350,
            'power_clean': 220,
            '40_yard': 4.5,
            '10_yard': 1.7,
            'vertical_jump': 35,
            'broad_jump': 90,
            '1_mile': 6.0,
            '5k_run': 22.0,
            'bench_press_power': 2.5,
            'squat_power': 3.5,
            'deadlift_power': 4.0,
            'power_clean_power': 3.0,
            'body_fat_percentage': 18.0,
            'skeletal_muscle_mass': 180,
            'quad_asymmetry': 0.8,
            'calf_asymmetry': 0.6,
            'hang_clean': 220,
            'L_drill': 6.5,
            'incline_bench_press_power': 2.5,
            'front_squat_power': 3.5,
            'romanian_deadlift_power': 4.0,
            'hang_clean_power': 3.0,
            'waist_hip_ratio': 0.9,
            'muscle_mass': 170,
            'quad_symmetry': 0.8,
            'calf_symmetry': 0.6
        }, (80, 89)),
        ('Unfit Individual', {
            'bench_press': 150,
            'squat': 200,
            'deadlift': 250,
            'power_clean': 150,
            '40_yard': 5.5,
            '10_yard': 2.0,
            'vertical_jump': 25,
            'broad_jump': 70,
            '1_mile': 7.0,
            '5k_run': 27.0,
            'bench_press_power': 2.0,
            'squat_power': 2.5,
            'deadlift_power': 3.0,
            'power_clean_power': 2.5,
            'body_fat_percentage': 20.0,
            'skeletal_muscle_mass': 150,
            'quad_asymmetry': 1.5,
            'calf_asymmetry': 1.0,
            'hang_clean': 150,
            'L_drill': 8.0,
            'incline_bench_press_power': 2.0,
            'front_squat_power': 2.5,
            'romanian_deadlift_power': 3.0,
            'hang_clean_power': 2.5,
            'waist_hip_ratio': 1.0,
            'muscle_mass': 140,
            'quad_symmetry': 1.5,
            'calf_symmetry': 1.0
        }, (20, 29)),
        ('Very Low Fitness Level', {
            'bench_press': 100,
            'squat': 150,
            'deadlift': 200,
            'power_clean': 100,
            '40_yard': 6.0,
            '10_yard': 2.5,
            'vertical_jump': 20,
            'broad_jump': 60,
            '1_mile': 8.0,
            '5k_run': 30.0,
            'bench_press_power': 1.5,
            'squat_power': 2.0,
            'deadlift_power': 2.5,
            'power_clean_power': 2.0,
            'body_fat_percentage': 25.0,
            'skeletal_muscle_mass': 120,
            'quad_asymmetry': 2.0,
            'calf_asymmetry': 1.5,
            'hang_clean': 100,
            'L_drill': 10.0,
            'incline_bench_press_power': 1.5,
            'front_squat_power': 2.0,
            'romanian_deadlift_power': 2.5,
            'hang_clean_power': 2.0,
            'waist_hip_ratio': 1.2,
            'muscle_mass': 120,
            'quad_symmetry': 2.0,
            'calf_symmetry': 1.5
        }, (10, 19))
    ]
    
    for label, data, expected_range in test_cases:
        save_user_data(data)
        load_and_process_data()
        kino_score = pd.read_csv('combined_data_w_score.csv').iloc[-1]['KinoScore']
        print(f'{label}: {kino_score} (Expected: {expected_range})')

test_kino_score()
"""
#if __name__ == '__main__':
    #main()                      
"""    
# Assuming you have loaded your data into a pandas DataFrame df
# Replace this with your actual data loading code
df1 = pd.read_csv('fabricated_v1_data.csv')
df2 = pd.read_csv('user_data.csv')
df1.drop(columns=['user_id'], inplace=True)
# Concatenate the dataframes
combined_df = pd.concat([df1, df2], ignore_index=True)
#print(combined_df)
# Define the metrics you want to include
metrics = [
    'bench_press', 'squat', 'deadlift', 'power_clean', 
    '40_yard', '10_yard', 'vertical_jump', 'broad_jump', 
    '1_mile', '5k_run', 'bench_press_power', 'squat_power', 
    'deadlift_power', 'power_clean_power', 'body_fat_percentage', 
    'skeletal_muscle_mass', 'quad_asymmetry', 'calf_asymmetry'
]

# Normalize each metric to 0-100 range
normalized_metrics = combined_df[metrics].apply(lambda x: (x - x.min()) / (x.max() - x.min()) * 100)
print(normalized_metrics)
# Calculate KinoScore
kino_score = normalized_metrics.sum(axis=1)/10
kino_score = kino_score.apply(lambda x: min(x, 100))

# Add KinoScore to the DataFrame if needed
combined_df['KinoScore'] = kino_score

# Print or return the DataFrame with KinoScore
print(combined_df[['KinoScore']].iloc[-1])  # Adjust columns as per your DataFrame structure


# Visualizations
# Histogram for normalized metrics
plt.figure(figsize=(12, 8))
normalized_metrics.hist(bins=20, layout=(6, 3), figsize=(15, 15))
plt.suptitle('Histograms of Normalized Metrics', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Heatmap for correlations
plt.figure(figsize=(14, 10))
corr_matrix = normalized_metrics.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Normalized Metrics', fontsize=16)
plt.show()

# Scatter plot for KinoScore distribution
plt.figure(figsize=(10, 6))
plt.scatter(combined_df.index, combined_df['KinoScore'], alpha=0.7)
plt.title('KinoScore Distribution', fontsize=16)
plt.xlabel('Index')
plt.ylabel('KinoScore')
plt.show()


new_user_df = pd.read_csv('user_data.csv')
# Normalize new user's data using the same min and max from the existing data
normalized_new_user_metrics = new_user_df[metrics].apply(
    lambda x: (x - combined_df[metrics].min()) / (combined_df[metrics].max() - combined_df[metrics].min()) * 100
)

# Calculate KinoScore for the new user
new_user_df['KinoScore'].iloc[-1] = normalized_new_user_metrics.sum(axis=1) / len(metrics)

# Print or return the new user's KinoScore
print(new_user_df[['KinoScore']])
"""

#def get_kino_score([metric]):
   # 100,150,200,100,6.0,2.5,20.0,60.0,8.0,30.0,1.5,2.0,2.5,2.0,25.0,120,2.0,1.5,100.0,10.0,1.5,2.0,2.5,2.0,1.2,120.0,2.0,1.5,15.35 = metric
    #return kinoiScore


# inialize github repo
# delete eerything you arent using (both .py scripts and csv files)
# put test cases on different file
# make it use only one csv (csv file gets update with new user data -> new csv file is used to make next predictions)