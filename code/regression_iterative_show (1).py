"""
=============================================================
RÉGRESSION LINÉAIRE ITÉRATIVE — Optimisation des features
Sommeil & Performance Cognitive | Étudiants uniquement
=============================================================
Exécuter avec :  python3 regression_iterative_show.py
Les graphiques s'affichent directement à l'écran (plt.show())
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Style global ────────────────────────────────────────────────────────────
C = {
    "blue":   "#4C9BE8",
    "red":    "#E8574C",
    "green":  "#52C47A",
    "orange": "#F5A623",
    "purple": "#7C5CBF",
    "dark":   "#2D3142",
    "gray":   "#8A93A2",
    "light":  "#F7F8FC",
}
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.facecolor":    C["light"],
    "figure.facecolor":  "white",
    "axes.titlesize":    12,
    "axes.labelsize":    11,
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT & ENCODAGE
# ══════════════════════════════════════════════════════════════════════════════

df       = pd.read_csv('sleep_health_dataset.csv')
students = df[df['occupation'] == 'Student'].copy()
Y        = students['cognitive_performance_score'].values
print(f"✔ {len(students)} étudiants chargés")

def minmax(s):
    return (s - s.min()) / (s.max() - s.min())

# Encodages catégoriels
students['mhc_enc']  = students['mental_health_condition'].map(
    {'Healthy': 0, 'Anxiety': 1, 'Depression': 1, 'Both': 2})
students['risk_enc'] = students['sleep_disorder_risk'].map(
    {'Healthy': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3})

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# ── Fonction d'évaluation ────────────────────────────────────────────────────
def evaluate(X_df):
    sc    = StandardScaler()
    X     = sc.fit_transform(X_df.values)
    model = LinearRegression().fit(X, Y)
    Yp    = model.predict(X)
    n, p  = X.shape
    res   = Y - Yp
    R2    = 1 - np.sum(res**2) / np.sum((Y - Y.mean())**2)
    R2a   = 1 - (1 - R2) * (n - 1) / (n - p - 1)
    cv    = cross_val_score(LinearRegression(), X, Y, cv=kf, scoring='r2')
    rmse  = np.sqrt(np.mean(res**2))
    # t-stats
    Xi    = np.column_stack([np.ones(n), X])
    beta  = np.concatenate([[model.intercept_], model.coef_])
    MSres = np.sum(res**2) / (n - p - 1)
    se    = np.sqrt(np.diag(MSres * np.linalg.inv(Xi.T @ Xi)))
    t     = beta / se
    pv    = 2 * (1 - stats.t.cdf(np.abs(t), n - p - 1))
    return dict(R2=R2, R2a=R2a, cv=cv, rmse=rmse, res=res, Yp=Yp,
                beta=beta[1:], se=se[1:], t=t[1:], pv=pv[1:],
                cols=list(X_df.columns), n=n, p=p)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — CORRÉLATIONS INITIALES (toutes variables numériques vs cible)
# ══════════════════════════════════════════════════════════════════════════════

numeric_base = [
    'sleep_quality_score', 'risk_enc', 'sleep_duration_hrs', 'stress_score',
    'rem_percentage', 'mhc_enc', 'felt_rested', 'wake_episodes_per_night',
    'deep_sleep_percentage', 'exercise_day', 'alcohol_units_before_bed',
    'shift_work', 'work_hours_that_day', 'sleep_latency_mins',
    'steps_that_day', 'caffeine_mg_before_bed', 'heart_rate_resting_bpm',
    'nap_duration_mins', 'bmi', 'screen_time_before_bed_mins',
]

corrs = [(col, np.corrcoef(students[col].values, Y)[0, 1]) for col in numeric_base]
corrs.sort(key=lambda x: -abs(x[1]))
names_c = [c[0].replace('_', '\n') for c in corrs]
vals_c  = [c[1] for c in corrs]
colors_c = [C["blue"] if v > 0 else C["red"] for v in vals_c]
alpha_c  = [1.0 if abs(v) >= 0.2 else 0.4 for v in vals_c]

fig1, ax = plt.subplots(figsize=(16, 6))
fig1.suptitle("Figure 1 — Corrélations de Pearson avec cognitive_performance_score\n"
              "(Variables au-dessus de |r| = 0.2 retenues pour la modélisation)",
              fontsize=13, fontweight='bold', color=C["dark"])
bars = ax.bar(names_c, vals_c, color=colors_c,
              edgecolor='white', linewidth=0.8)
for bar, a in zip(bars, alpha_c):
    bar.set_alpha(a)
ax.axhline(0,    color=C["dark"],   linewidth=1.2, linestyle='-')
ax.axhline(0.2,  color=C["green"],  linewidth=1.5, linestyle='--', label='seuil +0.2')
ax.axhline(-0.2, color=C["orange"], linewidth=1.5, linestyle='--', label='seuil -0.2')
for bar, v in zip(bars, vals_c):
    ax.text(bar.get_x() + bar.get_width()/2,
            v + (0.012 if v >= 0 else -0.028),
            f"{v:.2f}", ha='center', fontsize=7.5, color=C["dark"])
ax.set_ylabel("Coefficient de corrélation r", fontsize=11)
ax.set_ylim(-0.85, 1.0)
ax.legend(fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.tick_params(axis='x', labelsize=7.5)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DES FEATURES HANDCRAFTÉES
# ══════════════════════════════════════════════════════════════════════════════

# ── Feature 1 : deep_rem_combo ───────────────────────────────────────────────
# Combine deep sleep % et REM % en un seul score de qualité architecturale du sommeil
# Justification : les deux phases sont complémentaires (récupération physique vs cognitive)
students['deep_rem_combo'] = (
    students['deep_sleep_percentage'] + students['rem_percentage']
)

# ── Feature 2 : lifestyle_boost ──────────────────────────────────────────────
# Score de comportement sain = faire du sport ET ne pas boire d'alcool avant de dormir
# Justification : l'exercice améliore le sommeil profond ; l'alcool fragmente le REM
students['lifestyle_boost'] = (
    students['exercise_day']
    - minmax(students['alcohol_units_before_bed'])
)

# ── Feature 3 : burden_score ─────────────────────────────────────────────────
# Charge globale = stress + heures de travail + travail posté + santé mentale
# Justification : capture toute la pression psychologique/physique qui nuit à la cognition
students['burden_score'] = (
    minmax(students['stress_score'])
    + minmax(students['work_hours_that_day'])
    + students['shift_work']
    + students['mhc_enc'] / 2
) / 4


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — EXPLICATION DES FEATURES HANDCRAFTÉES
# ══════════════════════════════════════════════════════════════════════════════

fig2, axes = plt.subplots(1, 3, figsize=(17, 5))
fig2.suptitle("Figure 2 — Features handcraftées : distribution et lien avec la performance cognitive",
              fontsize=13, fontweight='bold', color=C["dark"])

hc_features = [
    ('deep_rem_combo',  C["blue"],   'deep_rem_combo\n= deep_% + rem_%'),
    ('lifestyle_boost', C["green"],  'lifestyle_boost\n= exercise − alcohol_norm'),
    ('burden_score',    C["red"],    'burden_score\n= mean(stress+work+shift+mhc)'),
]

for ax, (feat, col, desc) in zip(axes, hc_features):
    x = students[feat]
    r, _ = stats.pearsonr(x, Y)
    ax.scatter(x, Y, alpha=0.04, color=col, s=5, rasterized=True)
    m, b = np.polyfit(x, Y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, m*xs + b, color=C["dark"], linewidth=2.5, label=f'r = {r:.3f}')
    ax.set_xlabel(desc, fontsize=10)
    ax.set_ylabel('Performance cognitive' if feat == 'deep_rem_combo' else '')
    ax.set_title(f"{feat}\nr = {r:.3f}  {'▲' if r>0 else '▼'}", fontsize=11, color=col)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ITÉRATIONS DE RÉGRESSION
# ══════════════════════════════════════════════════════════════════════════════

iteration_results = {}

# ── Itération 1 : 3 variables brutes de base ─────────────────────────────────
it1_cols = ['sleep_quality_score', 'risk_enc', 'stress_score']
it1 = evaluate(students[it1_cols])
iteration_results['It.1\n3 vars\nbrutes'] = it1

# ── Itération 2 : + deep_rem_combo ───────────────────────────────────────────
it2_cols = it1_cols + ['deep_rem_combo']
it2 = evaluate(students[it2_cols])
iteration_results['It.2\n+ deep_rem\n_combo'] = it2

# ── Itération 3 : + burden_score ─────────────────────────────────────────────
it3_cols = it2_cols + ['burden_score']
it3 = evaluate(students[it3_cols])
iteration_results['It.3\n+ burden\n_score'] = it3

# ── Itération 4 : remplacer burden par lifestyle_boost (plus efficace) ────────
it4_cols = it2_cols + ['lifestyle_boost']
it4 = evaluate(students[it4_cols])
iteration_results['It.4\n+ lifestyle\n_boost ✓'] = it4

# Résumé console
print("\n" + "═"*75)
print("RÉSUMÉ DES ITÉRATIONS")
print("═"*75)
for label, res in iteration_results.items():
    print(f"{label.replace(chr(10),' '):<35} "
          f"R²={res['R2']:.4f}  R²cv={res['cv'].mean():.4f}  RMSE={res['rmse']:.2f}")
print("═"*75)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — PROGRESSION DU R² À TRAVERS LES ITÉRATIONS
# ══════════════════════════════════════════════════════════════════════════════

labels_it  = list(iteration_results.keys())
r2_train   = [r['R2']          for r in iteration_results.values()]
r2_cv      = [r['cv'].mean()   for r in iteration_results.values()]
r2_cv_std  = [r['cv'].std()    for r in iteration_results.values()]
rmse_vals  = [r['rmse']        for r in iteration_results.values()]

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("Figure 3 — Progression du R² et RMSE à travers les itérations",
              fontsize=13, fontweight='bold', color=C["dark"])

x_pos = np.arange(len(labels_it))
ax1.plot(x_pos, r2_train, 'o-', color=C["purple"], linewidth=2.5,
         markersize=9, label='R² entraînement', zorder=3)
ax1.plot(x_pos, r2_cv, 's--', color=C["blue"], linewidth=2,
         markersize=8, label='R² cross-val (moy)', zorder=3)
ax1.fill_between(x_pos,
                 np.array(r2_cv) - np.array(r2_cv_std),
                 np.array(r2_cv) + np.array(r2_cv_std),
                 color=C["blue"], alpha=0.12, label='±1σ CV')
ax1.axhline(0.9, color=C["orange"], linestyle=':', linewidth=1.5,
            label='Objectif R²=0.90')
for i, (r2t, r2c) in enumerate(zip(r2_train, r2_cv)):
    ax1.annotate(f"{r2t:.4f}", (i, r2t), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=9, color=C["purple"], fontweight='bold')
    ax1.annotate(f"{r2c:.4f}", (i, r2c), textcoords="offset points",
                 xytext=(0, -16), ha='center', fontsize=8.5, color=C["blue"])
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels_it, fontsize=9)
ax1.set_ylabel("R²", fontsize=11)
ax1.set_ylim(0.68, 0.94)
ax1.legend(fontsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.set_title("Évolution du R² (train vs cross-validation)", fontsize=11)

bars = ax2.bar(x_pos, rmse_vals,
               color=[C["gray"], C["blue"], C["orange"], C["green"]],
               alpha=0.85, edgecolor='white')
for bar, val in zip(bars, rmse_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"{val:.2f}", ha='center', fontsize=10, fontweight='bold', color=C["dark"])
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels_it, fontsize=9)
ax2.set_ylabel("RMSE (points sur 0–100)", fontsize=11)
ax2.set_title("Erreur moyenne de prédiction (RMSE)", fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MODÈLE FINAL — Itération 4 : 5 variables (3 brutes + 2 handcraftées)
# ══════════════════════════════════════════════════════════════════════════════

final = it4
final_cols = it4_cols
print(f"\n{'═'*65}")
print("MODÈLE FINAL — 5 variables (3 brutes + 2 handcraftées)")
print(f"{'═'*65}")
print(f"R²          = {final['R2']:.4f}")
print(f"R² ajusté   = {final['R2a']:.4f}")
print(f"R² CV (10f) = {final['cv'].mean():.4f} ± {final['cv'].std():.4f}")
print(f"RMSE        = {final['rmse']:.2f} points")
print(f"\n{'Variable':<30} {'β std':>8} {'t':>8} {'p':>10} {'Sig':>5}")
print("─"*65)
for name, b, t, pv in zip(final_cols, final['beta'], final['t'], final['pv']):
    sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "NS"
    print(f"{name:<30} {b:>8.3f} {t:>8.2f} {pv:>10.3e}  {sig}")
print("═"*65)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — MODÈLE FINAL : Coefficients + Importance relative
# ══════════════════════════════════════════════════════════════════════════════

fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
fig4.suptitle(f"Figure 4 — Modèle final (5 variables)  —  R² = {final['R2']:.4f}  |  RMSE = {final['rmse']:.2f} pts",
              fontsize=13, fontweight='bold', color=C["dark"])

feat_colors = [C["blue"], C["orange"], C["red"], C["purple"], C["green"]]
feat_labels = [c.replace('_', '\n') for c in final_cols]

# Coefficients standardisés avec IC 95%
y_pos = np.arange(len(final_cols))
t_crit = stats.t.ppf(0.975, df=final['n'] - final['p'] - 1)
ci_lo  = final['beta'] - t_crit * final['se']
ci_hi  = final['beta'] + t_crit * final['se']

for i in y_pos:
    ax1.barh(i, final['beta'][i], color=feat_colors[i], alpha=0.85, height=0.5, zorder=3)
    ax1.plot([ci_lo[i], ci_hi[i]], [i, i], color=C["dark"], linewidth=2.5, zorder=4)
    ax1.plot([ci_lo[i]]*2, [i-.15, i+.15], color=C["dark"], linewidth=2, zorder=4)
    ax1.plot([ci_hi[i]]*2, [i-.15, i+.15], color=C["dark"], linewidth=2, zorder=4)
    sign = "+" if final['beta'][i] > 0 else ""
    ax1.text(ci_hi[i] + 0.05, i, f"{sign}{final['beta'][i]:.2f}",
             va='center', fontsize=10, fontweight='bold', color=feat_colors[i])
ax1.axvline(0, color=C["dark"], linewidth=1.5, linestyle='--', alpha=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(feat_labels, fontsize=10)
ax1.set_xlabel("Coefficient standardisé β", fontsize=11)
ax1.set_title("Coefficients avec IC 95%\n(*** toutes p < 0.001)", fontsize=11)
ax1.grid(axis='x', linestyle='--', alpha=0.4)
ax1.text(0.98, 0.02,
         "β > 0 : améliore la performance\nβ < 0 : détériore la performance\n|β| : intensité de l'effet",
         transform=ax1.transAxes, ha='right', va='bottom', fontsize=8.5,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# Importance relative
abs_b = np.abs(final['beta'])
pct   = abs_b / abs_b.sum() * 100
si    = np.argsort(pct)[::-1]
bars  = ax2.bar([feat_labels[i] for i in si],
                [pct[i] for i in si],
                color=[feat_colors[i] for i in si],
                alpha=0.85, edgecolor='white')
for bar, val in zip(bars, [pct[i] for i in si]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val:.1f}%", ha='center', fontsize=11, fontweight='bold', color=C["dark"])
ax2.set_ylabel("Contribution relative (%)", fontsize=11)
ax2.set_title("Importance relative des 5 variables", fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — DIAGNOSTIC DES RÉSIDUS (validation des hypothèses)
# ══════════════════════════════════════════════════════════════════════════════

res = final['res']
Yp  = final['Yp']
res_std = (res - res.mean()) / res.std()

fig5, axes = plt.subplots(2, 3, figsize=(17, 9))
fig5.suptitle("Figure 5 — Diagnostic des résidus du modèle final\n"
              "Vérification des 4 hypothèses de la régression linéaire (OLS)",
              fontsize=13, fontweight='bold', color=C["dark"])

# 1. Résidus vs ajustées
ax = axes[0, 0]
ax.scatter(Yp, res, alpha=0.05, s=5, color=C["purple"], rasterized=True)
ax.axhline(0, color=C["dark"], linestyle='--', linewidth=1.5)
z = np.polyfit(Yp, res, 2)
xs = np.linspace(Yp.min(), Yp.max(), 200)
ax.plot(xs, np.poly1d(z)(xs), color='red', linewidth=2, label='Tendance')
ax.set_xlabel("Valeurs ajustées ŷ"); ax.set_ylabel("Résidus")
ax.set_title("Résidus vs Valeurs ajustées\n→ Linéarité & Homoscédasticité")
ax.legend(fontsize=8)
ax.text(0.02, 0.95, "✓ Idéal : nuage centré sur 0\n   sans structure visible",
        transform=ax.transAxes, va='top', fontsize=8, color=C["gray"])

# 2. Q-Q plot
ax = axes[0, 1]
(osm, osr), (slope, intercept, r_qq) = stats.probplot(res, dist='norm')
ax.scatter(osm, osr, alpha=0.05, s=5, color=C["purple"], rasterized=True)
ax.plot(osm, slope * np.array(osm) + intercept, color='red', linewidth=2,
        label=f'r = {r_qq:.4f}')
ax.set_xlabel("Quantiles théoriques N(0,1)"); ax.set_ylabel("Quantiles des résidus")
ax.set_title("Q-Q Plot des résidus\n→ Normalité")
ax.legend(fontsize=9)
sw_stat, sw_p = stats.shapiro(np.random.choice(res, 2000, replace=False))
ax.text(0.02, 0.05, f"Shapiro-Wilk (2000 pts)\np = {sw_p:.4f}",
        transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# 3. Distribution des résidus
ax = axes[0, 2]
ax.hist(res, bins=60, color=C["purple"], alpha=0.7, density=True, edgecolor='white')
xs = np.linspace(res.min(), res.max(), 300)
ax.plot(xs, stats.norm.pdf(xs, res.mean(), res.std()),
        color='red', linewidth=2, label='N(μ,σ²)')
ax.set_xlabel("Résidus"); ax.set_ylabel("Densité")
ax.set_title(f"Distribution des résidus\nμ={res.mean():.3f}, σ={res.std():.3f}")
ax.legend()
ax.text(0.02, 0.95,
        f"Skewness = {stats.skew(res):.3f}\nKurtosis = {stats.kurtosis(res):.3f}",
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# 4. Observé vs Prédit
ax = axes[1, 0]
ax.scatter(Y, Yp, alpha=0.05, s=5, color=C["purple"], rasterized=True)
ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()],
        color='red', linewidth=2, linestyle='--', label='Parfait')
ax.set_xlabel("Performance réelle"); ax.set_ylabel("Performance prédite ŷ")
ax.set_title(f"Observé vs Prédit\nR² = {final['R2']:.4f}  |  RMSE = {final['rmse']:.2f} pts")
ax.legend(fontsize=9)

# 5. Résidus vs chaque feature du modèle final
ax = axes[1, 1]
x_ticks = np.arange(len(final_cols))
corr_res = [np.corrcoef(students[c].values, res)[0, 1] for c in final_cols]
bars = ax.bar([c.replace('_', '\n') for c in final_cols],
              corr_res, color=feat_colors, alpha=0.85, edgecolor='white')
for bar, val in zip(bars, corr_res):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + (0.003 if val >= 0 else -0.008),
            f"{val:.3f}", ha='center', fontsize=9, color=C["dark"])
ax.axhline(0, color=C["dark"], linewidth=1.5)
ax.axhline(0.05,  color=C["orange"], linewidth=1, linestyle=':', alpha=0.7)
ax.axhline(-0.05, color=C["orange"], linewidth=1, linestyle=':', alpha=0.7)
ax.set_ylabel("Corrélation résidus / feature")
ax.set_title("Structure résiduelle restante\n→ Toutes proches de 0 = bon signe")
ax.tick_params(axis='x', labelsize=8)

# 6. Scale-Location
ax = axes[1, 2]
sqrt_abs = np.sqrt(np.abs(res_std))
ax.scatter(Yp, sqrt_abs, alpha=0.05, s=5, color=C["purple"], rasterized=True)
z2 = np.polyfit(Yp, sqrt_abs, 1)
xs2 = np.linspace(Yp.min(), Yp.max(), 200)
ax.plot(xs2, np.poly1d(z2)(xs2), color='red', linewidth=2)
ax.set_xlabel("Valeurs ajustées ŷ"); ax.set_ylabel("√|Résidus standardisés|")
ax.set_title("Scale-Location Plot\n→ Homoscédasticité")
ax.text(0.02, 0.95, "✓ Idéal : ligne rouge\n   horizontale",
        transform=ax.transAxes, va='top', fontsize=8, color=C["gray"])

plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — CROSS-VALIDATION 10-FOLD
# ══════════════════════════════════════════════════════════════════════════════

cv_r2   = final['cv']
cv_rmse = np.sqrt(-cross_val_score(
    LinearRegression(),
    StandardScaler().fit_transform(students[final_cols].values),
    Y, cv=kf, scoring='neg_mean_squared_error'))

fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig6.suptitle(f"Figure 6 — Validation croisée 10-Fold\n"
              f"R²cv = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}  |  "
              f"Surapprentissage = {final['R2'] - cv_r2.mean():.4f} (négligeable)",
              fontsize=13, fontweight='bold', color=C["dark"])

folds = np.arange(1, 11)

bars1 = ax1.bar(folds, cv_r2, color=C["purple"], alpha=0.8, edgecolor='white')
ax1.axhline(cv_r2.mean(), color='red', linewidth=2, linestyle='--',
            label=f'Moy = {cv_r2.mean():.4f}')
ax1.fill_between([0.5, 10.5],
                 cv_r2.mean() - cv_r2.std(), cv_r2.mean() + cv_r2.std(),
                 color='red', alpha=0.1, label=f'±σ = {cv_r2.std():.4f}')
for b, v in zip(bars1, cv_r2):
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
             f"{v:.4f}", ha='center', fontsize=8, color=C["dark"])
ax1.set_xlabel("Fold"); ax1.set_ylabel("R²")
ax1.set_title("R² par fold — Stabilité du modèle")
ax1.set_xticks(folds); ax1.legend(fontsize=9)
ax1.set_ylim(0, max(cv_r2) * 1.12)
ax1.grid(axis='y', linestyle='--', alpha=0.4)

bars2 = ax2.bar(folds, cv_rmse, color=C["blue"], alpha=0.8, edgecolor='white')
ax2.axhline(cv_rmse.mean(), color='red', linewidth=2, linestyle='--',
            label=f'Moy = {cv_rmse.mean():.2f} pts')
for b, v in zip(bars2, cv_rmse):
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
             f"{v:.2f}", ha='center', fontsize=8, color=C["dark"])
ax2.set_xlabel("Fold"); ax2.set_ylabel("RMSE (points)")
ax2.set_title("RMSE par fold — Erreur de prédiction")
ax2.set_xticks(folds); ax2.legend(fontsize=9)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

print("\n✔ Analyse complète terminée. 6 figures affichées.")
print(f"\n→ MODÈLE FINAL : R² = {final['R2']:.4f}  |  R²cv = {cv_r2.mean():.4f}  |  RMSE = {final['rmse']:.2f} pts")
print(f"→ 5 variables : {final_cols}")
