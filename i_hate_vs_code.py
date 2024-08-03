import csv
import os

user_data_dict = {
    "user_id": 1,
    "bench_press": 250, "incline_bench_press": 180.0, "squat": 300.0, "front_squat": 230.0,
    "deadlift": 400.0, "romanian_deadlift": 290.0, "power_clean": 150.0, "hang_clean": 140.0,
    "40_yard": 4.5, "20_yard_shuttle": 4.2, "10_yard": 1.6, "L_drill": 7.8,
    "vertical_jump": 30.0, "100_meter": 12.0, "broad_jump": 110.0, "60_meter": 8.0,
    "1_mile": 7.5, "marathon": 240.0, "5k_run": 20.0, "half_marathon": 120.0,
    "bench_press_power": 1000.0, "incline_bench_press_power": 950.0, "squat_power": 1500.0, "front_squat_power": 1400.0,
    "deadlift_power": 1600.0, "romanian_deadlift_power": 1900.0, "power_clean_power": 1200.0, "hang_clean_power": 1300.0,
    "body_fat_percentage": 15.0, "waist_hip_ratio": 0.9, "skeletal_muscle_mass": 40.0, "muscle_mass": 50.0,
    "quad_asymmetry": 5.0, "quad_symmetry": 95.0, "calf_asymmetry": 1.30, "calf_symmetry": 3,
    "sleep_quality": 7, "energy_levels": 6, "stress_levels": 4, "mood": 5, "focus": 6,
    "pain_levels": 2, "consistency": 7, "steps": 10000, "walking_distance": 8, "flights_climbed": 12,
    "max_heart_rate": 180, "resting_heart_rate": 60, "heart_rate_variability": 50,
    "calories": 2500, "protein": 150, "carbohydrates": 300, "fats": 70, "hydration": 3
}

def append_dict_to_csv(dictionary, filename):
    # Check if the file exists
    file_exists = os.path.isfile(filename)
    
    # Open the file in append mode
    with open(filename, 'a', newline='') as csvfile:
        # Create a csv writer object
        csvwriter = csv.writer(csvfile)
        
        # If the file does not exist, write the header
        if not file_exists:
            header = dictionary.keys()
            csvwriter.writerow(header)
        
        # Write the dictionary values
        csvwriter.writerow(dictionary.values())

# Specify the filename
filename = 'historical_data.csv'

# Call the function to append the dictionary to the CSV file
append_dict_to_csv(user_data_dict, filename)

print(f'Dictionary values have been appended to {filename}')
