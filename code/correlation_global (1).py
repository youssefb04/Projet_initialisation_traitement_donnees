import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('sleep_health_dataset.csv')
students = df[df['occupation'] == 'Student'].copy()

# ============================================================
# ENCODAGE DES VARIABLES CATÉGORIELLES
# ============================================================
students_enc = students.copy()

students_enc['day_type_enc']        = students['day_type'].map({'Weekday': 0, 'Weekend': 1})
students_enc['gender_enc']          = students['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
students_enc['chronotype_enc']      = students['chronotype'].map({'Morning': 0, 'Neutral': 1, 'Evening': 2})
students_enc['mental_health_enc']   = students['mental_health_condition'].map({'Healthy': 0, 'Anxiety': 1, 'Depression': 2, 'Both': 3})
students_enc['sleep_disorder_enc']  = students['sleep_disorder_risk'].map({'Healthy': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3})
students_enc['season_enc']          = students['season'].map({'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3})

# ============================================================
# TOUTES LES VARIABLES
# ============================================================
ALL_VARS = [
    'age', 'bmi',
    'sleep_duration_hrs', 'sleep_quality_score', 'rem_percentage',
    'deep_sleep_percentage', 'sleep_latency_mins', 'wake_episodes_per_night',
    'stress_score', 'exercise_day', 'steps_that_day', 'work_hours_that_day',
    'screen_time_before_bed_mins', 'caffeine_mg_before_bed',
    'alcohol_units_before_bed', 'shift_work', 'nap_duration_mins',
    'sleep_aid_used', 'weekend_sleep_diff_hrs',
    'heart_rate_resting_bpm', 'room_temperature_celsius', 'felt_rested',
    'chronotype_enc', 'mental_health_enc', 'sleep_disorder_enc',
    'day_type_enc', 'season_enc', 'gender_enc'
]

# ============================================================
# CALCUL DES CORRÉLATIONS
# ============================================================
corr = students_enc[ALL_VARS + ['cognitive_performance_score']]\
       .corr()['cognitive_performance_score']\
       .drop('cognitive_performance_score')\
       .sort_values()

print('=== Corrélation de toutes les variables avec cognitive_performance_score ===')
print(f'{"Variable":<35} {"r":>8}')
print('-' * 45)
for var, r in corr.items():
    print(f'{var:<35} {r:>+.4f}')

# ============================================================
# VISUALISATION
# ============================================================
colors = ['#d73027' if r < 0 else '#1a9850' for r in corr.values]

fig, ax = plt.subplots(figsize=(10, 12))
ax.barh(corr.index, corr.values, color=colors, edgecolor='white', height=0.7)
ax.axvline(0, color='black', linewidth=0.8, alpha=0.3)

for i, (var, r) in enumerate(corr.items()):
    ax.text(r + (0.01 if r >= 0 else -0.01), i,
            f'{r:+.2f}', va='center',
            ha='left' if r >= 0 else 'right', fontsize=8)

ax.set_title('Corrélation de toutes les variables avec\ncognitive_performance_score (étudiants)',
             fontsize=13, pad=15)
ax.set_xlabel('Corrélation de Pearson (r)', fontsize=11)
ax.set_xlim(-1.0, 1.1)

plt.tight_layout()
plt.savefig('fig_all_correlations.png', dpi=150, bbox_inches='tight')
plt.show()