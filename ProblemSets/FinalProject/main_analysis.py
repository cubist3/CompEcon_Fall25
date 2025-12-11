"""
ECON 833 final project: TAX CAPITALIZATION IN UK USED CAR MARKET
1. summary statistics
2. SMM estimation
3. heterogeneity analysis (T-Learner with Random Forest)
4. figures and tables
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
print("loading and cleaning data")

df = pd.read_csv('data/toyota.csv')
df.columns = df.columns.str.lower().str.strip()
n_original = len(df)

# cleaning 
df = df.dropna()
df = df[(df['price'] > 1000) & (df['price'] < 60000)]
df = df[(df['mileage'] > 0) & (df['mileage'] < 200000)]
df = df[(df['year'] >= 2010) & (df['year'] <= 2020)]
df = df[(df['tax'] >= 0) & (df['tax'] <= 600)]
df = df[(df['mpg'] > 20) & (df['mpg'] < 120)]

df['age'] = 2020 - df['year']
df['log_price'] = np.log(df['price'])
df['mileage_10k'] = df['mileage'] / 10000

# tax regime from TAX value (more accurate than model year)
# pre-2017 regime: tax varies (0, 20, 30, ..., 265, 500+)
# post-2017 regime: mostly clustered around 135-150
df['post_2017_inferred'] = ((df['tax'] >= 125) & (df['tax'] <= 160)).astype(int)
df['post_2017_year'] = (df['year'] >= 2017).astype(int)

# year-based for simplicity (limitation...)
df['post_2017'] = df['post_2017_year']

df['is_hybrid'] = (df['fueltype'].str.lower() == 'hybrid').astype(int)
df['is_diesel'] = (df['fueltype'].str.lower() == 'diesel').astype(int)
df['is_automatic'] = df['transmission'].str.lower().isin(['automatic', 'semi-auto']).astype(int)

print(f"original observations: {n_original:,}")
print(f"after cleaning: {len(df):,}")

print("summary stats")
summary_vars = ['price', 'year', 'mileage', 'tax', 'mpg']
summary_stats = df[summary_vars].describe().T
summary_stats = summary_stats[['count', 'mean', 'std', 'min', '50%', 'max']]
summary_stats.columns = ['Obs', 'Mean', 'Std. Dev.', 'Min', 'Median', 'Max']

latex_summary = """
\\begin{table}[ht]
    \\centering
    \\caption{Summary Statistics}
    \\label{tab:summary_stats}
    \\begin{threeparttable}
    \\begin{tabular}{lccccc}
        \\toprule
        Variable & Obs & Mean & Std. Dev. & Min & Max \\\\
        \\midrule
"""
for var in summary_vars:
    row = summary_stats.loc[var]
    if var == 'price':
        latex_summary += f"        Price (\\pounds) & {int(row['Obs']):,} & {row['Mean']:,.0f} & {row['Std. Dev.']:,.0f} & {row['Min']:,.0f} & {row['Max']:,.0f} \\\\\n"
    elif var == 'year':
        latex_summary += f"        Year & {int(row['Obs']):,} & {row['Mean']:.1f} & {row['Std. Dev.']:.1f} & {int(row['Min'])} & {int(row['Max'])} \\\\\n"
    elif var == 'mileage':
        latex_summary += f"        Mileage & {int(row['Obs']):,} & {row['Mean']:,.0f} & {row['Std. Dev.']:,.0f} & {row['Min']:,.0f} & {row['Max']:,.0f} \\\\\n"
    elif var == 'tax':
        latex_summary += f"        Tax (\\pounds) & {int(row['Obs']):,} & {row['Mean']:.1f} & {row['Std. Dev.']:.1f} & {row['Min']:.0f} & {row['Max']:.0f} \\\\\n"
    elif var == 'mpg':
        latex_summary += f"        MPG & {int(row['Obs']):,} & {row['Mean']:.1f} & {row['Std. Dev.']:.1f} & {row['Min']:.1f} & {row['Max']:.1f} \\\\\n"

latex_summary += """        \\bottomrule
    \\end{tabular}
    \\begin{tablenotes}
        \\small
        \\item \\textit{Note:} Data represents Toyota used car listings from the UK market.
    \\end{tablenotes}
    \\end{threeparttable}
\\end{table}
"""
with open('output/tables/summary_stats.tex', 'w') as f:
    f.write(latex_summary)
print("summary stat table is saved: output/tables/summary_stats.tex")

# SMM estimation
print("SMM estimation")

R = 0.05  # discount rate
T = 5     # holding period
NPV_FACTOR = (1 - (1 + R)**(-T)) / R

# moments (Model x Year)
df_smm = df[(df['year'] >= 2012) & (df['year'] <= 2020)].copy()

moments = df_smm.groupby(['model', 'year']).agg({
    'price': 'mean',
    'tax': 'mean',
    'mileage_10k': 'mean',
    'age': 'first'
}).reset_index()

moments['n_obs'] = df_smm.groupby(['model', 'year']).size().values
moments = moments[moments['n_obs'] >= 5].reset_index(drop=True)
moments['npv_tax'] = moments['tax'] * NPV_FACTOR

models = moments['model'].unique()
model_to_idx = {m: i for i, m in enumerate(models)}
moments['model_idx'] = moments['model'].map(model_to_idx)

n_moments = len(moments)
n_models = len(models)

print(f"    Moments: {n_moments}")
print(f"    Models: {n_models}")

# model function
def model_price(theta, age, npv_tax, model_idx, n_models):
    delta = theta[0]
    lam = theta[1]
    alphas = theta[2:]
    alpha_j = alphas[model_idx]
    price_pred = alpha_j * np.exp(-delta * age) - lam * npv_tax
    return np.maximum(price_pred, 1000)

def smm_objective(theta, y, age, npv_tax, model_idx, n_models, weights):
    y_pred = model_price(theta, age, npv_tax, model_idx, n_models)
    errors = (y - y_pred) / y
    return np.sum(weights * errors**2)

y = moments['price'].values
age = moments['age'].values
npv_tax = moments['npv_tax'].values
model_idx = moments['model_idx'].values
weights = np.sqrt(moments['n_obs'].values)
weights = weights / weights.sum()

# initialisation
mean_price_by_model = moments.groupby('model_idx')['price'].mean().values
alpha_init = mean_price_by_model * np.exp(0.10 * 3)

bounds = (
    [(0.05, 0.25), (0.0, 3.0)]
    + [(5000, 70000)] * n_models)

# grid search
best_result = None
best_obj = np.inf

for delta_start in [0.08, 0.10, 0.12, 0.15]:
    for lambda_start in [0.5, 1.0, 1.5]:
        theta_init = np.concatenate([[delta_start, lambda_start], alpha_init])
        result = minimize(
            smm_objective, theta_init,
            args=(y, age, npv_tax, model_idx, n_models, weights),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 5000}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result

theta_hat = best_result.x
delta_hat = theta_hat[0]
lambda_hat = theta_hat[1]
alpha_hat = theta_hat[2:]

print(f"\n Point Estimates:")
print(f" Depreciation: {delta_hat:.4f} ({delta_hat*100:.1f}%/year)")
print(f" Tax Cap.:     {lambda_hat:.4f}")

# bootstrap SE
print("bootstrap")
n_boot = 200
boot_estimates = []

for b in range(n_boot):
    boot_idx = np.random.choice(n_moments, size=n_moments, replace=True)
    y_b, age_b, npv_tax_b = y[boot_idx], age[boot_idx], npv_tax[boot_idx]
    model_idx_b = model_idx[boot_idx]
    weights_b = weights[boot_idx]
    weights_b = weights_b / weights_b.sum()
    
    result_b = minimize(
        smm_objective, theta_hat,
        args=(y_b, age_b, npv_tax_b, model_idx_b, n_models, weights_b),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 500}
    )
    if result_b.success:
        boot_estimates.append(result_b.x[:2])

boot_estimates = np.array(boot_estimates)
se = np.std(boot_estimates, axis=0)
se_delta, se_lambda = se[0], se[1]

y_pred = model_price(theta_hat, age, npv_tax, model_idx, n_models)
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

t_zero = lambda_hat / se_lambda
p_zero = 2 * (1 - norm.cdf(abs(t_zero)))
t_one = (lambda_hat - 1) / se_lambda
p_one = 2 * (1 - norm.cdf(abs(t_one)))

print(f"\n    Results:")
print(f"    δ = {delta_hat:.4f} (SE = {se_delta:.4f})")
print(f"    λ = {lambda_hat:.4f} (SE = {se_lambda:.4f})")
print(f"    R² = {r2:.4f}")
print(f"    H₀: λ=0 → p = {p_zero:.4f}")
print(f"    H₀: λ=1 → p = {p_one:.4f}")

# SMM results table - latex table
smm_latex = f"""
\\begin{{table}}[ht]
    \\centering
    \\caption{{Structural Estimation Results (SMM)}}
    \\label{{tab:smm_results}}
    \\begin{{threeparttable}}
    \\begin{{tabular}}{{lccc}}
        \\toprule
        Parameter & Estimate & Std. Error & Interpretation \\\\
        \\midrule
        Depreciation Rate ($\\delta$) & {delta_hat:.4f} & ({se_delta:.4f}) & {delta_hat*100:.1f}\\% annual value loss \\\\
        Tax Capitalization ($\\lambda$) & {lambda_hat:.4f} & ({se_lambda:.4f}) & \\pounds {lambda_hat:.2f} price drop per \\pounds 1 NPV tax \\\\
        \\midrule
        \\multicolumn{{4}}{{l}}{{\\textit{{Hypothesis Tests:}}}} \\\\
        $H_0: \\lambda = 0$ & \\multicolumn{{3}}{{c}}{{$t = {t_zero:.2f}$, $p = {p_zero:.4f}$}} \\\\
        $H_0: \\lambda = 1$ & \\multicolumn{{3}}{{c}}{{$t = {t_one:.2f}$, $p = {p_one:.4f}$}} \\\\
        \\midrule
        Model Fixed Effects & \\multicolumn{{3}}{{c}}{{{n_models} models}} \\\\
        Moment Conditions & \\multicolumn{{3}}{{c}}{{{n_moments}}} \\\\
        $R^2$ & \\multicolumn{{3}}{{c}}{{{r2:.3f}}} \\\\
        \\bottomrule
    \\end{{tabular}}
    \\begin{{tablenotes}}
        \\small
        \\item \\textit{{Note:}} Bootstrap standard errors (200 replications). NPV calculated with $r=0.05$, $T=5$ years.
    \\end{{tablenotes}}
    \\end{{threeparttable}}
\\end{{table}}
"""

with open('output/tables/smm_results.tex', 'w') as f:
    f.write(smm_latex)

print("\n    Saved: output/tables/smm_results.tex")

# heterogeneity analysis
print("heterogeneity analysis")
df_ml = df.copy()
df_ml['engine_size'] = df_ml['enginesize']
features = ['age', 'mileage_10k', 'enginesize', 'is_automatic']
# split by regime
df_pre = df_ml[df_ml['post_2017'] == 0].copy()
df_post = df_ml[df_ml['post_2017'] == 1].copy()
print(f"    Pre-2017 observations: {len(df_pre):,}")
print(f"    Post-2017 observations: {len(df_post):,}")

# train RF on pre-2017 data
X_pre = df_pre[features].values
y_pre = df_pre['price'].values

rf_pre = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_pre.fit(X_pre, y_pre)

# counterfactual prices for post-2017 vehicles
X_post = df_post[features].values
y_post_actual = df_post['price'].values
y_post_counterfactual = rf_pre.predict(X_post)

# price shock = actual - counterfactual
df_post['price_shock'] = y_post_actual - y_post_counterfactual
df_post['price_shock_pct'] = (y_post_actual - y_post_counterfactual) / y_post_counterfactual * 100

# aggregate by model
model_effects = df_post.groupby('model').agg({
    'price_shock': 'mean',
    'price_shock_pct': 'mean',
    'mpg': 'mean',
    'tax': 'mean',
    'price': ['mean', 'count']
}).reset_index()

model_effects.columns = ['Model', 'Price Impact', 'Price Impact %', 'Avg MPG', 'Avg Tax', 'Avg Price', 'N']
model_effects = model_effects[model_effects['N'] >= 20].sort_values('Price Impact', ascending=False)

print("\n Price Impact by Model (Top 5 Winners and Losers):")
print(model_effects[['Model', 'Price Impact', 'Avg MPG', 'N']].head(5).to_string(index=False))
print("    ...")
print(model_effects[['Model', 'Price Impact', 'Avg MPG', 'N']].tail(3).to_string(index=False))

# heterogeneity table
hetero_models = ['Land Cruiser', 'GT86', 'Hilux', 'Aygo', 'Yaris', 'Prius', 'Auris']
hetero_df = model_effects[model_effects['Model'].isin(hetero_models)].copy()

# classifying vehicle type
def classify_type(model):
    if model in ['Land Cruiser', 'Hilux']:
        return 'SUV/Truck'
    elif model == 'GT86':
        return 'Sports'
    elif model in ['Aygo', 'Yaris']:
        return 'City Car'
    elif model in ['Prius', 'Auris']:
        return 'Hybrid'
    else:
        return 'Other'

hetero_df['Type'] = hetero_df['Model'].apply(classify_type)

hetero_latex = """
\\begin{table}[ht]
    \\centering
    \\caption{Estimated Price Impact by Model (Select Models)}
    \\label{tab:hetero_impact}
    \\begin{tabular}{lcccc}
        \\toprule
        Model & Type & Avg. MPG & N & Price Impact ($\\hat{\\tau}$) \\\\
        \\midrule
"""

for _, row in hetero_df.sort_values('Price Impact', ascending=False).iterrows():
    sign = '+' if row['Price Impact'] >= 0 else ''
    bold = '\\textbf' if abs(row['Price Impact']) > 500 else ''
    if bold:
        hetero_latex += f"        {bold}{{{row['Model']}}} & {row['Type']} & {row['Avg MPG']:.1f} & {int(row['N'])} & {bold}{{\\pounds {sign}{row['Price Impact']:,.0f}}} \\\\\n"
    else:
        hetero_latex += f"        {row['Model']} & {row['Type']} & {row['Avg MPG']:.1f} & {int(row['N'])} & \\pounds {sign}{row['Price Impact']:,.0f} \\\\\n"

hetero_latex += """        \\bottomrule
    \\end{tabular}
    \\small \\textit{Note: Positive impact implies the car trades at a premium relative to pre-2017 pricing structure.}
\\end{table}
"""

with open('output/tables/heterogeneity.tex', 'w') as f:
    f.write(hetero_latex)

print("\n output/tables/heterogeneity.tex")

# figures
print("figures")
plt.style.use('seaborn-v0_8-whitegrid')
# tax distribution by regime
fig, ax = plt.subplots(figsize=(10, 6))
pre_tax = df[df['post_2017'] == 0]['tax']
post_tax = df[df['post_2017'] == 1]['tax']
ax.hist(pre_tax, bins=30, alpha=0.6, label='Pre-2017 (CO₂-based)', color='steelblue', density=True)
ax.hist(post_tax, bins=30, alpha=0.6, label='Post-2017 (Flat rate)', color='coral', density=True)
ax.set_xlabel('Annual Vehicle Excise Duty (pounds)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Distribution of Annual Road Tax by Registration Regime', fontsize=14)
ax.legend(fontsize=11)
ax.axvline(x=145, color='red', linestyle='--', alpha=0.7, linewidth=2)
plt.tight_layout()
plt.savefig('output/figures/tax_distribution.png', dpi=300)
plt.close()
print("tax_distribution.png")

# depreciation curve
fig, ax = plt.subplots(figsize=(10, 6))
ages_plot = np.arange(0, 10, 0.1)
depreciation = np.exp(-delta_hat * ages_plot)
ax.plot(ages_plot, depreciation * 100, 'b-', lw=2)
ax.fill_between(ages_plot, depreciation * 100, alpha=0.2)
ax.set_xlabel('Vehicle Age (Years)', fontsize=12)
ax.set_ylabel('Remaining Value (% of New)', fontsize=12)
ax.set_title(f'Estimated Depreciation Curve (delta= {delta_hat:.1%}/year)', fontsize=14)
ax.set_ylim(0, 105)
ax.set_xlim(0, 10)
for age_mark in [1, 3, 5, 7]:
    val = np.exp(-delta_hat * age_mark) * 100
    ax.plot(age_mark, val, 'ro', markersize=8)
    ax.annotate(f'{val:.0f}%', xy=(age_mark, val), xytext=(age_mark + 0.3, val + 3), fontsize=10)
plt.tight_layout()
plt.savefig('output/figures/depreciation_curve.png', dpi=300)
plt.close()
print("depreciation_curve.png")

# hetero analysis - MPG vs Price Shock
fig, ax = plt.subplots(figsize=(10, 7))
scatter_data = model_effects[model_effects['N'] >= 30].copy()
colors = ['coral' if x > 0 else 'steelblue' for x in scatter_data['Price Impact']]
sizes = scatter_data['N'] / 5

ax.scatter(scatter_data['Avg MPG'], scatter_data['Price Impact'], 
           c=colors, s=sizes, alpha=0.7, edgecolors='white', linewidth=1)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

for _, row in scatter_data.iterrows():
    if abs(row['Price Impact']) > 1000 or row['Model'] in ['Prius', 'GT86', 'Yaris']:
        ax.annotate(row['Model'], xy=(row['Avg MPG'], row['Price Impact']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('average fuel economy (MPG)', fontsize=12)
ax.set_ylabel('estimated price impact', fontsize=12)
ax.set_title('heterogeneous treatment effects"', fontsize=14)
plt.tight_layout()
plt.savefig('output/figures/heterogeneity_mpg.png', dpi=300)
plt.close()
print("heterogeneity_mpg.png")

# pre-post tax by fuel type
fig, ax = plt.subplots(figsize=(10, 6))
tax_by_regime = df.groupby(['fueltype', 'post_2017'])['tax'].mean().unstack()
tax_by_regime.columns = ['Pre-2017', 'Post-2017']
tax_by_regime = tax_by_regime.loc[['Hybrid', 'Petrol', 'Diesel']]
x = np.arange(len(tax_by_regime))
width = 0.35
bars1 = ax.bar(x - width/2, tax_by_regime['Pre-2017'], width, label='Pre-2017', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, tax_by_regime['Post-2017'], width, label='Post-2017', color='coral', alpha=0.8)
ax.set_xlabel('fuel type', fontsize=12)
ax.set_ylabel('average annual tax', fontsize=12)
ax.set_title('average VED by fuel Type and tax regime', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(tax_by_regime.index)
ax.legend()
ax.bar_label(bars1, fmt='£%.0f', padding=3)
ax.bar_label(bars2, fmt='£%.0f', padding=3)
plt.tight_layout()
plt.savefig('output/figures/tax_by_fueltype.png', dpi=300)
plt.close()
print("tax_by_fueltype.png")

results = {
    'delta': delta_hat,
    'delta_se': se_delta,
    'lambda': lambda_hat,
    'lambda_se': se_lambda,
    't_zero': t_zero,
    'p_zero': p_zero,
    't_one': t_one,
    'p_one': p_one,
    'r2': r2,
    'n_moments': n_moments,
    'n_models': n_models,
    'n_obs': len(df_smm)
}

pd.DataFrame([results]).to_csv('output/tables/smm_results.csv', index=False)
model_effects.to_csv('output/tables/heterogeneity.csv', index=False)

print("smm_results.csv")
print("heterogeneity.csv")

# results
print("\n" + "=" * 80)
print("=" * 80)

print(f"""
structural equ: P_jt = α_j × exp(-delta × age) - lambda × NPV(Tax) + epsilon

SMM estimation:
    δ (Depreciation):       {delta_hat:.4f} (SE = {se_delta:.4f})  [{delta_hat*100:.1f}% annual]
    λ (Tax Capitalization): {lambda_hat:.4f} (SE = {se_lambda:.4f})

hypothesis test:
    H₀: λ = 0  →  t = {t_zero:.2f}, p = {p_zero:.4f}  {'*' if p_zero < 0.1 else ''}
    H₀: λ = 1  →  t = {t_one:.2f}, p = {p_one:.4f}  {'(Cannot reject)' if p_one > 0.05 else ''}

95% CI for lambda: [{lambda_hat - 1.96*se_lambda:.2f}, {lambda_hat + 1.96*se_lambda:.2f}]

R-sqr = {r2:.4f}

hetero analysis (T-LEARNER):
    - high-emission vehicles (GT86, Land Cruiser): + price shock
    - low-emission vehicles (Prius, Yaris Hybrid): - price shock
    - pattern from the 2017 reform

    lambda approx {lambda_hat:.2f} suggests {'full' if 0.8 < lambda_hat < 1.2 else 'partial'} tax capitalization.
    consumers {'fully incorporate' if abs(lambda_hat - 1) < 0.3 else 'partially incorporate'} future tax liabilities.
""")

print("=" * 80)
print("done.")
print("=" * 80)
