import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('sleep_health_dataset.csv')
students = df[df['occupation'] == 'Student'].copy()

# Variables catégorielles à analyser
CAT_VARS = {
    'mental_health_condition': ['Healthy', 'Anxiety', 'Depression', 'Both'],
    'sleep_disorder_risk':     ['Healthy', 'Mild', 'Moderate', 'Severe'],
    'day_type':                ['Weekday', 'Weekend'],
    'chronotype':              ['Morning', 'Neutral', 'Evening'],
    'season':                  ['Spring', 'Summer', 'Autumn', 'Winter'],
    'gender':                  ['Female', 'Male', 'Other'],
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax, (var, categories) in zip(axes, CAT_VARS.items()):

    # Distribution conditionnelle P(cognitive_score | var = categorie)
    data_by_cat = [
        students[students[var] == cat]['cognitive_performance_score'].dropna().values
        for cat in categories
    ]

    # Boxplot
    bp = ax.boxplot(data_by_cat, labels=categories, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))

    colors = ['#1a9850', '#f0a500', '#d73027', '#5b9bd5',
              '#a9d18e', '#8B0000']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Moyenne par catégorie
    means = [np.mean(d) for d in data_by_cat]
    ax.plot(range(1, len(categories) + 1), means,
            'D', color='black', markersize=6, zorder=5, label='Moyenne')

    # Annoter les moyennes
    for i, mean in enumerate(means):
        ax.text(i + 1, mean + 1.5, f'{mean:.1f}',
                ha='center', fontsize=9, fontweight='bold')

    # Test statistique
    from scipy import stats
    if len(data_by_cat) == 2:
        stat, p = stats.ttest_ind(data_by_cat[0], data_by_cat[1])
        test_name = 't-test'
    else:
        stat, p = stats.f_oneway(*data_by_cat)
        test_name = 'ANOVA'

    significance = '★ Significatif' if p < 0.05 else '✗ Non significatif'
    ax.set_title(f'P(cognitive_score | {var})\n{test_name}: p = {p:.4f}  {significance}',
                 fontsize=10)
    ax.set_xlabel(var, fontsize=9)
    ax.set_ylabel('cognitive_performance_score', fontsize=9)
    ax.tick_params(axis='x', rotation=15)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)

plt.suptitle('Probabilité conditionnelle P(cognitive_performance_score | variable catégorielle)\nPopulation étudiante',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('fig_conditional_probability.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# RÉSUMÉ STATISTIQUE
# ============================================================
print('=== RÉSUMÉ : Probabilité conditionnelle ===\n')
for var, categories in CAT_VARS.items():
    print(f'--- {var} ---')
    for cat in categories:
        subset = students[students[var] == cat]['cognitive_performance_score']
        print(f'  P(cognitive_score | {var} = {cat}): '
              f'mean = {subset.mean():.2f}, std = {subset.std():.2f}, n = {len(subset)}')

    data_by_cat = [students[students[var] == cat]['cognitive_performance_score'].values
                   for cat in categories]
    if len(categories) == 2:
        _, p = stats.ttest_ind(data_by_cat[0], data_by_cat[1])
        print(f'  → t-test p-value = {p:.4f} : {"Significatif ✓" if p < 0.05 else "Non significatif ✗"}')
    else:
        _, p = stats.f_oneway(*data_by_cat)
        print(f'  → ANOVA p-value = {p:.4f} : {"Significatif ✓" if p < 0.05 else "Non significatif ✗"}')
    print()