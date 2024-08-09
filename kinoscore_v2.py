import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import csv
import os

metrics = [
    'bench_press', 'squat', 'deadlift', 'power_clean', '40_yard', '10_yard', 
    'vertical_jump', 'broad_jump', '1_mile', '5k_run', 'bench_press_power', 'squat_power', 
    'deadlift_power', 'power_clean_power', 'body_fat_percentage', 'skeletal_muscle_mass', 
    'quad_asymmetry', 'calf_asymmetry', 'muscle_mass', 'waist_hip_ratio',
    'marathon', 'half_marathon', 'sleep_quality', 'energy_levels', 'stress_levels', 
    'mood', 'focus', 'pain_levels', 'consistency', 'steps', 'walking_distance', 
    'flights_climbed', 'max_heart_rate', 'resting_heart_rate', 'heart_rate_variability', 
    'calories', 'protein', 'carbohydrates', 'fats', 'hydration'
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

#which metrics should be inverted (lower is better)
invert_metrics = [
    '40_yard', '20_yard_shuttle', '10_yard', 'L_drill', 
    '100_meter', '60_meter', '1_mile', '5k_run', 
    'marathon', 'half_marathon', 'body_fat_percentage', 'waist_hip_ratio',
    'quad_asymmetry', 'calf_asymmetry', 'quad_symmetry', 'calf_symmetry',
    'resting_heart_rate', 'stress_levels', 'pain_levels'
]

#goals and their associated metrics
goals = {
    'Lose excess body fat': ['body_fat_percentage', 'waist_hip_ratio'],
    'Build muscle mass': ['skeletal_muscle_mass', 'muscle_mass'],
    'Improve endurance': ['1_mile', '5k_run', 'marathon', 'half_marathon'],
    'Stick to my diet': [None],
    'Improve my sleep schedule': [None],
    'Get faster': ['40_yard', '10_yard', '100_meter'],
    'Tone down': ['body_fat_percentage', 'skeletal_muscle_mass', 'muscle_mass'],
    'Become more flexible': [None],
    'Get more powerful': ['vertical_jump', 'broad_jump', 'bench_press_power', 'squat_power', 'deadlift_power', 'power_clean_power'],
    'Drink more water': [None],
    'Improve sleep quality': ['sleep_quality'],
    'Increase energy levels': ['energy_levels'],
    'Reduce stress': ['stress_levels'],
    'Enhance mood': ['mood'],
    'Boost focus': ['focus'],
    'Decrease pain levels': ['pain_levels'],
    'Improve workout consistency': ['consistency'],
    'Increase daily steps': ['steps'],
    'Increase walking distance': ['walking_distance'],
    'Increase flights climbed': ['flights_climbed'],
    'Improve max heart rate': ['max_heart_rate'],
    'Improve resting heart rate': ['resting_heart_rate'],
    'Improve heart rate variability': ['heart_rate_variability'],
    'Optimize nutrition': ['calories', 'protein', 'carbohydrates', 'fats'],
    'Improve hydration': ['hydration']
}

historical_lengths = {
    'sleep_quality': 'week',
    'resting_heart_rate': 'week',
    'heart_rate_variability': 'week',
    'calories': 'week',
    'stress_levels': 'week',
    'energy_levels': 'week',
    'mood': 'week',
    'focus': 'week',
    'pain_levels': 'month',
    'consistency': 'month',
    'max_heart_rate': 'day',
    'steps': 'week',
    'walking_distance': 'week',
    'flights_climbed': 'week',
    'feeling_of_progress': 'month',
    'average_progress': 'month',
    'bench_press': 'year',
    'squat': 'year',
    'deadlift': 'year',
    'power_clean': 'year',
    '40_yard': 'year',
    '10_yard': 'year',
    'vertical_jump': 'year',
    'broad_jump': 'year',
    '1_mile': 'year',
    '5k_run': 'year',
    'bench_press_power': 'year',
    'squat_power': 'year',
    'deadlift_power': 'year',
    'power_clean_power': 'year',
    'body_fat_percentage': 'year',
    'skeletal_muscle_mass': 'year',
    'quad_asymmetry': 'year',
    'calf_asymmetry': 'year',
    'muscle_mass': 'year',
    'waist_hip_ratio': 'year',
    'marathon': 'year',
    'half_marathon': 'year'
}

def perform_pca(df):
    standardized_df = (df - df.mean()) / df.std()
    pca = PCA()
    pca.fit(standardized_df.dropna())
    explained_variance = pca.explained_variance_ratio_
    metric_weights = {metric: explained_variance[i] for i, metric in enumerate(df.columns)}
    return metric_weights

def get_historical_average(df, metric, period):
    if metric not in df.columns or df[metric].empty:
        return np.nan
    if period == 'day':
        return df[metric].iloc[-1]
    elif period == 'week':
        if len(df[metric]) < 7:
            return df[metric].mean()  # Use available data
        return df[metric].rolling(window=7).mean().iloc[-1]
    elif period == 'month':
        if len(df[metric]) < 30:
            return df[metric].mean()  
        return df[metric].rolling(window=30).mean().iloc[-1]
    elif period == 'year':
        if len(df[metric]) < 365:
            return df[metric].mean()  
        return df[metric].rolling(window=365).mean().iloc[-1]
    else:
        return df[metric].mean()

def calculate_trend_value(current_value, historical_average):
    return current_value / historical_average if historical_average else np.nan

def calculate_weighted_historical_average(df, metric, weights):
    if metric not in df.columns or df[metric].empty:
        return np.nan

    weighted_average = 0
    total_weight = 0

    for days_ago, weight in weights.items():
        if len(df[metric]) > days_ago:
            weighted_average += df[metric].iloc[-days_ago] * weight
            total_weight += weight

    if total_weight == 0:
        return np.nan

    return weighted_average / total_weight

def normalize_and_invert(value, min_value, max_value, invert=False, trend=None, weight_distribution=None, improvement_penalty=False):
    if pd.isna(value) or pd.isna(min_value) or pd.isna(max_value):
        return np.nan

    
    normalized_value = (value - min_value) / (max_value - min_value)
    
    # Invert normalization if needed
    if invert:
        normalized_value = 1 - normalized_value
    
    # Combine with trend if provided
    if trend:
        trend_weight = weight_distribution['trend'] if weight_distribution else 0.5
        objective_weight = weight_distribution['objective'] if weight_distribution else 0.5
        trend_normalized = trend.get('current', 0) / trend.get('historical', 1)
        normalized_value = (objective_weight * normalized_value) + (trend_weight * trend_normalized)
    
    
    if improvement_penalty and trend:
        if value == trend['historical']:
            normalized_value *= 0.9  # Apply a 10% penalty for lack of improvement
    
    # Ensure the combined normalized value is within the range of 0 to 1
    normalized_value = np.clip(normalized_value, 0, 1)
    
    return normalized_value

metrics_weight_distribution = {
    'bench_press': {'objective': 0, 'trend': 1},
    'squat': {'objective': 0, 'trend': 1},
    'deadlift': {'objective': 0, 'trend': 1},
    'power_clean': {'objective': 0, 'trend': 1},
    '40_yard': {'objective': 0.5, 'trend': 0.5},
    '10_yard': {'objective': 0.5, 'trend': 0.5},
    'vertical_jump': {'objective': 0.5, 'trend': 0.5},
    'broad_jump': {'objective': 0.5, 'trend': 0.5},
    '1_mile': {'objective': 0.5, 'trend': 0.5},
    '5k_run': {'objective': 0.5, 'trend': 0.5},
    'bench_press_power': {'objective': 0, 'trend': 1},
    'squat_power': {'objective': 0, 'trend': 1},
    'deadlift_power': {'objective': 0, 'trend': 1},
    'power_clean_power': {'objective': 0, 'trend': 1},
    'body_fat_percentage': {'objective': 0.5, 'trend': 0.5},
    'skeletal_muscle_mass': {'objective': 0.5, 'trend': 0.5},
    'quad_asymmetry': {'objective': 0.5, 'trend': 0.5},
    'calf_asymmetry': {'objective': 0.5, 'trend': 0.5},
    'muscle_mass': {'objective': 0.5, 'trend': 0.5},
    'waist_hip_ratio': {'objective': 0.5, 'trend': 0.5},
    'marathon': {'objective': 0.5, 'trend': 0.5},
    'half_marathon': {'objective': 0.5, 'trend': 0.5},
    'sleep_quality': {'objective': 0.5, 'trend': 0.5},
    'energy_levels': {'objective': 0.5, 'trend': 0.5},
    'stress_levels': {'objective': 0.5, 'trend': 0.5},
    'mood': {'objective': 0.5, 'trend': 0.5},
    'focus': {'objective': 0.5, 'trend': 0.5},
    'pain_levels': {'objective': 0.5, 'trend': 0.5},
    'consistency': {'objective': 0.5, 'trend': 0.5},
    'steps': {'objective': 0.5, 'trend': 0.5},
    'walking_distance': {'objective': 0.5, 'trend': 0.5},
    'flights_climbed': {'objective': 0.5, 'trend': 0.5},
    'max_heart_rate': {'objective': 0.5, 'trend': 0.5},
    'resting_heart_rate': {'objective': 0.5, 'trend': 0.5},
    'heart_rate_variability': {'objective': 0.5, 'trend': 0.5},
    'calories': {'objective': 0.5, 'trend': 0.5},
    'protein': {'objective': 0.5, 'trend': 0.5},
    'carbohydrates': {'objective': 0.5, 'trend': 0.5},
    'fats': {'objective': 0.5, 'trend': 0.5},
    'hydration': {'objective': 0.5, 'trend': 0.5}
}

def append_dict_to_csv(dictionary, filename):
    file_exists = os.path.isfile(filename)
    
    if file_exists:
        existing_df = pd.read_csv(filename)
        new_row = pd.DataFrame([dictionary])
        if not new_row.isin(existing_df).all(axis=None).all():
            # Find a metric to increment
            metric_to_increment = next(iter(dictionary.keys()))  # or choose a specific metric
            dictionary[metric_to_increment] += 0.1
            new_row = pd.DataFrame([dictionary])
        existing_df = pd.concat([existing_df, new_row], ignore_index=True)
        existing_df.to_csv(filename, index=False)
    else:
        with open(filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            header = dictionary.keys()
            csvwriter.writerow(header)
            csvwriter.writerow(dictionary.values())

def remove_duplicates_from_csv(filename):
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        df.drop_duplicates(keep='last', inplace=True)
        df.to_csv(filename, index=False)

def calculate_consistency_score(user_data, historical_df):
    weekly_weights = {
        1: 0.70,
        2: 0.20,
        3: 0.10,
        4: 0.10,
        5: 0.10,
        6: 0.10,
        7: 0.10
    }

    consistency_score = 0
    metric_count = 0

    
    active_minutes = user_data.get('active_minutes')
    if active_minutes is not None:
        historical_avg_active_minutes = calculate_weighted_historical_average(historical_df, 'active_minutes', weekly_weights)
        trend_value = calculate_trend_value(active_minutes, historical_avg_active_minutes)
        active_minutes_objective_score = np.interp(active_minutes, [0, 40, 60, 90, 120, 150], [10, 30, 50, 70, 90, 100])
        active_minutes_trend_score = np.interp(trend_value, [0.8, 0.9, 1.0, 1.1, 1.2], [20, 60, 80, 90, 100])
        active_minutes_score = 0.4 * active_minutes_objective_score + 0.6 * active_minutes_trend_score
        consistency_score += 0.4 * active_minutes_score
        metric_count += 1

    
    calories_burned = user_data.get('calories_burned')
    if calories_burned is not None:
        historical_avg_calories_burned = calculate_weighted_historical_average(historical_df, 'calories_burned', weekly_weights)
        trend_value = calculate_trend_value(calories_burned, historical_avg_calories_burned)
        calories_burned_objective_score = np.interp(calories_burned, [0, 1000, 1250, 1500, 2000, 2500], [10, 30, 50, 70, 90, 100])
        calories_burned_trend_score = np.interp(trend_value, [0.8, 0.9, 1.0, 1.1, 1.2], [20, 60, 80, 90, 100])
        calories_burned_score = 0.5 * calories_burned_objective_score + 0.5 * calories_burned_trend_score
        consistency_score += 0.3 * calories_burned_score
        metric_count += 1

    
    steps = user_data.get('steps')
    if steps is not None:
        historical_avg_steps = calculate_weighted_historical_average(historical_df, 'steps', weekly_weights)
        trend_value = calculate_trend_value(steps, historical_avg_steps)
        steps_objective_score = np.interp(steps, [0, 2000, 4000, 6000, 8000, 10000], [10, 30, 50, 70, 90, 100])
        steps_trend_score = np.interp(trend_value, [0.8, 0.9, 1.0, 1.1, 1.2], [20, 60, 80, 90, 100])
        steps_score = 0.5 * steps_objective_score + 0.5 * steps_trend_score
        consistency_score += 0.3 * steps_score
        metric_count += 1

    return consistency_score

def calculate_average_progress_score(user_data, historical_df):
    
    strength_metrics = ['bench_press', 'squat', 'deadlift', 'power_clean']
    speed_metrics = ['40_yard', '100_meter']
    endurance_metrics = ['1_mile', '5k_run', 'marathon', 'half_marathon']
    power_metrics = ['bench_press_power', 'squat_power', 'deadlift_power', 'power_clean_power']
    reps_metrics = ['bench_press', 'squat', 'deadlift', 'power_clean']
    
    
    strength_progress = calculate_strength_progress(user_data, historical_df, strength_metrics)
    speed_progress = calculate_speed_progress(user_data, historical_df, speed_metrics)
    endurance_progress = calculate_endurance_progress(user_data, historical_df, endurance_metrics)
    power_progress = calculate_power_progress(user_data, historical_df, power_metrics)
    reps_progress = calculate_reps_progress(user_data, historical_df, reps_metrics)
    
    
    average_progress_score = (strength_progress * 0.2 +
                              speed_progress * 0.2 +
                              endurance_progress * 0.15 +
                              power_progress * 0.2 +
                              reps_progress * 0.25)
    
    return average_progress_score

def calculate_strength_progress(user_data, historical_df, metrics):
    progress_score = 0
    metric_count = 0
    
    for metric in metrics:
        if metric in user_data:
            period = historical_lengths.get(metric, 'year')
            historical_average = get_historical_average(historical_df, metric, period)
            trend_value = calculate_trend_value(user_data[metric], historical_average)
            score = calculate_strength_score(user_data[metric], trend_value)
            progress_score += score
            metric_count += 1
    
    if metric_count > 0:
        progress_score /= metric_count
    
    return progress_score

def calculate_strength_score(current, trend):
    # Calculate strength score based on the trend
    if trend >= 1.10:
        return 100
    elif 1.07 <= trend < 1.10:
        return 94
    elif 1.04 <= trend < 1.07:
        return 85
    elif 1.01 <= trend < 1.04:
        return 75
    elif 1.00 <= trend < 1.01:
        return 70
    elif 0.95 <= trend < 1.00:
        return 60
    elif 0.90 <= trend < 0.95:
        return 50
    elif 0.80 <= trend < 0.90:
        return 20
    else:
        return 0

def calculate_speed_progress(user_data, historical_df, metrics):
    progress_score = 0
    metric_count = 0
    
    for metric in metrics:
        if metric in user_data:
            period = historical_lengths.get(metric, 'year')
            historical_average = get_historical_average(historical_df, metric, period)
            trend_value = calculate_trend_value(user_data[metric], historical_average)
            score = calculate_speed_score(user_data[metric], trend_value)
            progress_score += score
            metric_count += 1
    
    if metric_count > 0:
        progress_score /= metric_count
    
    return progress_score

def calculate_speed_score(current, trend):
    # Calculate speed score based on the trend
    if trend >= 1.03:
        return 100
    elif 1.02 <= trend < 1.03:
        return 94
    elif 1.01 <= trend < 1.02:
        return 80
    elif 1.00 <= trend < 1.01:
        return 65
    elif 0.99 <= trend < 1.00:
        return 50
    elif 0.97 <= trend < 0.99:
        return 40
    else:
        return 20

def calculate_endurance_progress(user_data, historical_df, metrics):
    progress_score = 0
    metric_count = 0
    
    for metric in metrics:
        if metric in user_data:
            period = historical_lengths.get(metric, 'year')
            historical_average = get_historical_average(historical_df, metric, period)
            trend_value = calculate_trend_value(user_data[metric], historical_average)
            score = calculate_endurance_score(user_data[metric], trend_value)
            progress_score += score
            metric_count += 1
    
    if metric_count > 0:
        progress_score /= metric_count
    
    return progress_score

def calculate_endurance_score(current, trend):
    # Calculate endurance score based on the trend
    if trend >= 1.10:
        return 100
    elif 1.07 <= trend < 1.10:
        return 92
    elif 1.04 <= trend < 1.07:
        return 82
    elif 1.01 <= trend < 1.04:
        return 70
    elif 1.00 <= trend < 1.01:
        return 60
    elif 0.95 <= trend < 1.00:
        return 45
    elif 0.90 <= trend < 0.95:
        return 30
    else:
        return 15

def calculate_power_progress(user_data, historical_df, metrics):
    progress_score = 0
    metric_count = 0
    
    for metric in metrics:
        if metric in user_data:
            period = historical_lengths.get(metric, 'year')
            historical_average = get_historical_average(historical_df, metric, period)
            trend_value = calculate_trend_value(user_data[metric], historical_average)
            score = calculate_power_score(user_data[metric], trend_value)
            progress_score += score
            metric_count += 1
    
    if metric_count > 0:
        progress_score /= metric_count
    
    return progress_score

def calculate_power_score(current, trend):
    # Calculate power score based on the trend
    if trend >= 1.10:
        return 100
    elif 1.07 <= trend < 1.10:
        return 94
    elif 1.04 <= trend < 1.07:
        return 85
    elif 1.01 <= trend < 1.04:
        return 75
    elif 1.00 <= trend < 1.01:
        return 70
    elif 0.95 <= trend < 1.00:
        return 60
    elif 0.90 <= trend < 0.95:
        return 50
    elif 0.80 <= trend < 0.90:
        return 20
    else:
        return 0

def calculate_reps_progress(user_data, historical_df, metrics):
    progress_score = 0
    metric_count = 0
    
    for metric in metrics:
        if metric in user_data:
            period = historical_lengths.get(metric, 'year')
            historical_average = get_historical_average(historical_df, metric, period)
            trend_value = calculate_trend_value(user_data[metric], historical_average)
            score = calculate_reps_score(user_data[metric], trend_value)
            progress_score += score
            metric_count += 1
    
    if metric_count > 0:
        progress_score /= metric_count
    
    return progress_score

def calculate_reps_score(current, trend):
    # Calculate reps score based on the trend
    if trend >= 1.20:
        return 100
    elif 1.15 <= trend < 1.20:
        return 92
    elif 1.10 <= trend < 1.15:
        return 83
    elif 1.05 <= trend < 1.10:
        return 70
    elif 1.00 <= trend < 1.05:
        return 60
    elif 0.95 <= trend < 1.00:
        return 40
    elif 0.90 <= trend < 0.95:
        return 30
    else:
        return 20

def calculate_heart_score(user_data, historical_df):
    historical_weights = {
        1: 0.35,
        2: 0.25,
        3: 0.25,
        4: 0.20,
        5: 0.20,
        6: 0.20,
        7: 0.20,
        14: 0.10,
        30: 0.10
    }

    heart_score = 0

    
    hrv = user_data.get('heart_rate_variability')
    if hrv is not None:
        historical_avg_hrv = calculate_weighted_historical_average(historical_df, 'heart_rate_variability', historical_weights)
        trend_value = calculate_trend_value(hrv, historical_avg_hrv)
        hrv_objective_score = np.interp(hrv, [20, 30, 40, 50, 60, 70], [10, 20, 40, 60, 80, 100])
        hrv_trend_score = np.interp(trend_value, [0.8, 0.9, 1.0, 1.05, 1.1], [10, 60, 80, 90, 100])
        hrv_score = 0.4 * hrv_objective_score + 0.6 * hrv_trend_score
        heart_score += 0.7 * hrv_score

    
    rhr = user_data.get('resting_heart_rate')
    if rhr is not None:
        historical_avg_rhr = calculate_weighted_historical_average(historical_df, 'resting_heart_rate', historical_weights)
        trend_value = calculate_trend_value(rhr, historical_avg_rhr)
        rhr_objective_score = np.interp(rhr, [50, 60, 70, 80, 90, 100], [100, 90, 80, 60, 40, 10])
        rhr_trend_score = np.interp(trend_value, [1.2, 1.1, 1.0, 0.95, 0.90], [20, 40, 80, 90, 100])
        rhr_score = 0.4 * rhr_objective_score + 0.6 * rhr_trend_score
        heart_score += 0.3 * rhr_score

    return heart_score

def calculate_recovery_score(user_data, historical_df):
    sleep_weights = {
        1: 0.40,
        2: 0.30,
        3: 0.20,
        4: 0.20,
        5: 0.20,
        6: 0.20,
        7: 0.20,
        14: 0.10
    }

    recovery_score = 0
    metric_count = 0

    
    sleep_quality = user_data.get('sleep_quality')
    if sleep_quality is not None:
        historical_avg_sleep_quality = calculate_weighted_historical_average(historical_df, 'sleep_quality', sleep_weights)
        trend_value = calculate_trend_value(sleep_quality, historical_avg_sleep_quality)
        sleep_quality_objective_score = np.interp(sleep_quality, [1, 2, 3, 4, 5], [10, 30, 50, 70, 100])
        sleep_quality_trend_score = np.interp(trend_value, [0.8, 0.9, 1.0, 1.1, 1.2], [20, 60, 80, 90, 100])
        sleep_quality_score = 0.4 * sleep_quality_objective_score + 0.6 * sleep_quality_trend_score
        recovery_score += 0.6 * sleep_quality_score
        metric_count += 1

    
    heart_score = calculate_heart_score(user_data, historical_df)
    recovery_score += 0.4 * heart_score
    metric_count += 1

    return recovery_score

def calculate_subjective_scores(user_data):
    recovery_score = (
        0.45 * user_data['refreshed_feeling'] +
        0.15 * user_data['workout_difficulty'] +
        0.30 * user_data['nutrition_sufficiency']
    )

    wellness_score = (
        0.40 * user_data['energy_level'] +
        0.40 * user_data['stress_management'] +
        0.20 * user_data['mood'] +
        0.10 * user_data['pain_levels']
    )

    avg_progress_score = user_data['progress_towards_goal']

    consistency_score = user_data['workout_consistency']

    subjective_scores = {
        'recovery_score': recovery_score,
        'wellness_score': wellness_score,
        'avg_progress_score': avg_progress_score,
        'consistency_score': consistency_score
    }

    return subjective_scores

def calculate_general_score(objective_score, subjective_scores):
    subjective_score = 0
    weight_sum = 0

    if subjective_scores['recovery_score'] is not None:
        subjective_score += 0.40 * subjective_scores['recovery_score']
        weight_sum += 0.40
    
    
    if subjective_scores['wellness_score'] is not None:
        subjective_score += 0.20 * subjective_scores['wellness_score']
        weight_sum += 0.20
    
    
    if subjective_scores['avg_progress_score'] is not None:
        subjective_score += 0.20 * subjective_scores['avg_progress_score']
        weight_sum += 0.20
    
    
    if subjective_scores['consistency_score'] is not None:
        subjective_score += 0.20 * subjective_scores['consistency_score']
        weight_sum += 0.20
    
    # Adjust the subjective score proportionally based on the weight_sum
    if weight_sum > 0:
        subjective_score /= weight_sum

    general_score = 0.60 * objective_score + 0.40 * subjective_score
    return general_score


def calculate_objective_score(user_data, historical_df):
    objective_score = 0
    metric_count = 0

    for metric in metrics:
        if metric in user_data:
            period = historical_lengths.get(metric, 'year')
            historical_average = get_historical_average(historical_df, metric, period)

            # Skip the metric if historical average is NaN or zero
            if pd.isna(historical_average) or historical_average == 0:
                continue

            # Check if historical data has valid min and max values
            min_value = historical_df[metric].min()
            max_value = historical_df[metric].max()
            if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
                continue

            trend_value = calculate_trend_value(user_data[metric], historical_average)
            weight_distribution = metrics_weight_distribution.get(metric, {'objective': 0.5, 'trend': 0.5})

            normalized_score = normalize_and_invert(
                user_data[metric],
                min_value,
                max_value,
                invert=metric in invert_metrics,
                trend={'current': user_data[metric], 'historical': historical_average},
                weight_distribution=weight_distribution
            )

            # Ensure the normalized score is valid before adding to the objective score
            if not pd.isna(normalized_score):
                objective_score += normalized_score
                metric_count += 1

    if metric_count > 0:
        objective_score /= metric_count
    else:
        objective_score = None  # Return None if no valid metrics were found

    return objective_score

def calculate_goal_centered_scores(user_data, selected_goals, primary_df, secondary_df, historical_df):
    goal_scores = {}
    user_history = historical_df
    
    for goal in selected_goals:
        if goal in goals:
            goal_metrics = goals[goal]
            goal_score = 0
            metric_count = 0
            
            for metric in goal_metrics:
                if metric and metric in user_data:
                    if metric in primary_df.columns:
                        min_value = primary_df[metric].min()
                        max_value = primary_df[metric].max()
                        period = historical_lengths.get(metric, 'year')
                        historical_average = get_historical_average(user_history, metric, period)
                        trend_value = calculate_trend_value(user_data[metric], historical_average)
                        weight_distribution = metrics_weight_distribution.get(metric, {'objective': 0.5, 'trend': 0.5})
                        normalized_score = normalize_and_invert(
                            user_data[metric], min_value, max_value, invert=metric in invert_metrics, trend={'current': user_data[metric], 'historical': historical_average}, weight_distribution=weight_distribution, improvement_penalty=True
                        )
                    elif metric in secondary_df.columns:
                        min_value = secondary_df[metric].min()
                        max_value = secondary_df[metric].max()
                        period = historical_lengths.get(metric, 'year')
                        historical_average = get_historical_average(user_history, metric, period)
                        trend_value = calculate_trend_value(user_data[metric], historical_average)
                        weight_distribution = metrics_weight_distribution.get(metric, {'objective': 0.5, 'trend': 0.5})
                        normalized_score = normalize_and_invert(
                            user_data[metric], min_value, max_value, invert=metric in invert_metrics, trend={'current': user_data[metric], 'historical': historical_average}, weight_distribution=weight_distribution, improvement_penalty=True
                        )
                    else:
                        normalized_score = np.nan

                    if not pd.isna(normalized_score):
                        goal_score += normalized_score
                        metric_count += 1
            
            if metric_count > 0:
                goal_scores[goal] = goal_score / metric_count
            else:
                goal_scores[goal] = np.nan

    return goal_scores

def kino_score_v2(user_data, selected_goals, metrics, primary_file='fabricated_v1_data.csv', secondary_file='secondary_metrics.csv', historical_file='historical_data.csv'):
    primary_df = pd.read_csv(primary_file)
    secondary_df = pd.read_csv(secondary_file)
    
    
    append_dict_to_csv(user_data, historical_file)
    
    historical_df = pd.read_csv(historical_file)
    
    primary_weights = perform_pca(primary_df)
    secondary_weights = perform_pca(secondary_df)
    
    user_data = {k: (float(v) if v not in ('n/a', None) else np.nan) for k, v in user_data.items()}
    
    for primary, secondary in secondary_metrics.items():
        if primary in user_data and pd.isna(user_data[primary]) and secondary in user_data:
            user_data[primary] = user_data[secondary]

    goal_weights = {goal: 1/len(selected_goals) for goal in selected_goals}
    
    metric_goal_weights = {}
    for goal in selected_goals:
        if goal in goals:
            goal_metrics = goals[goal]
            weight_per_metric = goal_weights[goal] / len(goal_metrics) if goal_metrics else 0
            for metric in goal_metrics:
                if metric:
                    if metric in metric_goal_weights:
                        metric_goal_weights[metric] += weight_per_metric
                    else:
                        metric_goal_weights[metric] = weight_per_metric

    primary_user_data = {metric: user_data[metric] for metric in metrics if metric in user_data}
    
    normalized_user_data = {}
    user_history = historical_df
    
    for metric in primary_user_data:
        if metric in primary_df.columns:
            min_value = primary_df[metric].min()
            max_value = primary_df[metric].max()
            period = historical_lengths.get(metric, 'year')
            historical_average = get_historical_average(user_history, metric, period)
            trend_value = calculate_trend_value(primary_user_data[metric], historical_average)
            weight_distribution = metrics_weight_distribution.get(metric, {'objective': 0.5, 'trend': 0.5})
            normalized_user_data[metric] = normalize_and_invert(
                primary_user_data[metric], min_value, max_value, invert=metric in invert_metrics, trend={'current': primary_user_data[metric], 'historical': historical_average}, weight_distribution=weight_distribution, improvement_penalty=True
            )
        elif metric in secondary_df.columns:
            min_value = secondary_df[metric].min()
            max_value = secondary_df[metric].max()
            period = historical_lengths.get(metric, 'year')
            historical_average = get_historical_average(user_history, metric, period)
            trend_value = calculate_trend_value(primary_user_data[metric], historical_average)
            weight_distribution = metrics_weight_distribution.get(metric, {'objective': 0.5, 'trend': 0.5})
            normalized_user_data[metric] = normalize_and_invert(
                primary_user_data[metric], min_value, max_value, invert=metric in invert_metrics, trend={'current': primary_user_data[metric], 'historical': historical_average}, weight_distribution=weight_distribution, improvement_penalty=True
            )
    
    scaled_user_data = {metric: normalized_user_data[metric] * 100 for metric in normalized_user_data}
    scaled_user_data = {metric: np.clip(value, 1, 100) for metric, value in scaled_user_data.items()}
    
    total_goal_weight = sum(metric_goal_weights.values())
    goal_adjusted_weights = {metric: metric_goal_weights.get(metric, 0) for metric in metrics}
    
    total_pca_weight = sum(primary_weights.values()) + sum(secondary_weights.values())
    pca_adjusted_weights = {metric: (primary_weights.get(metric, 0) + secondary_weights.get(metric, 0)) / total_pca_weight for metric in metrics}
    
    final_weights = {metric: pca_adjusted_weights[metric] + goal_adjusted_weights[metric] for metric in metrics}
    
    weighted_sum = 0
    weights_data = []
    print("Metrics contributing to KinoScore and their weights:")
    for metric in metrics:
        if metric in scaled_user_data:
            weight = final_weights.get(metric, 0)
            weights_data.append((metric, weight))
            print(f"Metric: {metric}, Weight: {weight:.2f}")
            weighted_sum += scaled_user_data[metric] * weight

    weighted_kino_score = weighted_sum / sum(final_weights.values())
    weighted_kino_score = round(weighted_kino_score, 2)
    
    # Ensure KinoScore is within 1-100 range
    weighted_kino_score = max(1, min(weighted_kino_score, 100))

    
    goal_centered_scores = calculate_goal_centered_scores(user_data, selected_goals, primary_df, secondary_df, historical_df)
    goal_centered_score = np.nanmean(list(goal_centered_scores.values()))

    
    final_kino_score = 0.5 * weighted_kino_score + 0.5 * goal_centered_score

    # Create radar chart data
    radar_chart_data = {
        'Recovery': calculate_recovery_score(user_data, historical_df),
        'Wellness': calculate_subjective_scores(user_data)['wellness_score'],
        'Average Progress': calculate_average_progress_score(user_data, historical_df),
        'Consistency': calculate_consistency_score(user_data, historical_df)
    }

    # Normalize radar chart data to 1-5 scale
    radar_chart_data = {k: (v / 100) * 5 for k, v in radar_chart_data.items()}

    # Plot radar chart
    labels = list(radar_chart_data.keys())
    values = list(radar_chart_data.values())

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.show()

    return final_kino_score, scaled_user_data

def generate_spider_chart(scores):
    labels = list(scores.keys())
    values = list(scores.values())

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.show()

# Example user data
user_data = {
    'active_minutes': 120,
    'calories_burned': 3500,
    'steps': 15000,
    'bench_press': 315,
    'squat': 405,
    'deadlift': 500,
    'power_clean': 275,
    '40_yard': 4.3,
    '100_meter': 10.5,
    '1_mile': 5.0,
    '5k_run': 18.0,
    'bench_press_power': 320,
    'squat_power': 415,
    'deadlift_power': 510,
    'power_clean_power': 280,
    'body_fat_percentage': 8,
    'skeletal_muscle_mass': 65,
    'quad_asymmetry': 2,
    'calf_asymmetry': 2,
    'muscle_mass': 75,
    'waist_hip_ratio': 0.85,
    'marathon': 200,
    'half_marathon': 90,
    'sleep_quality': 8,
    'energy_levels': 9,
    'stress_levels': 3,
    'mood': 8,
    'focus': 8,
    'pain_levels': 2,
    'consistency': 9,
    'steps': 15000,
    'walking_distance': 7,
    'flights_climbed': 15,
    'max_heart_rate': 190,
    'resting_heart_rate': 50,
    'heart_rate_variability': 65,
    'calories': 3500,
    'protein': 200,
    'carbohydrates': 400,
    'fats': 90,
    'hydration': 4,
    'refreshed_feeling': 9,
    'sleep_quality': 8,
    'workout_difficulty': 7,
    'nutrition_sufficiency': 8,
    'energy_level': 9,
    'stress_management': 8,
    'anxiety_levels': 2,
    'self_care': 7,
    'mood': 8,
    'pain_levels': 2,
    'progress_towards_goal': 8,
    'workout_consistency': 9
}


selected_goals = ['Lose excess body fat', 'Build muscle mass']
kino_score, scaled_user_data = kino_score_v2(user_data, selected_goals, metrics)
print("Final Kino Score:", kino_score)

# Normalize scores to a 1-5 scale for spider chart
normalized_scores = {key: np.interp(value, [1, 100], [1, 5]) for key, value in scaled_user_data.items()}

objective_score = calculate_objective_score(user_data, pd.read_csv('historical_data.csv'))
subjective_scores = calculate_subjective_scores(user_data)
general_score = calculate_general_score(objective_score, subjective_scores)

normalized_scores['General Score'] = np.interp(general_score, [1, 100], [1, 5])
normalized_scores.update({f'Subjective {key}': np.interp(value, [1, 100], [1, 5]) for key, value in subjective_scores.items()})

generate_spider_chart(normalized_scores)













