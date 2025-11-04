# income_inequality_full_extended.py
# Extended Income Inequality Analysis
# Includes: descriptive stats, histogram, boxplot, violin plot, Gini, correlation, regression, percentiles, hypothesis tests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
# --- 1. Data Generation ---
np.random.seed(42)
n = 500
# Lognormal incomes for Rural and Urban
rural = np.random.lognormal(mean=11.5, sigma=0.4, size=n)
urban = np.random.lognormal(mean=12.3, sigma=0.7, size=n)
rural, urban = np.round(rural), np.round(urban)
# Combine into a DataFrame
df = pd.concat([
    pd.DataFrame({'group': 'Rural', 'income': rural}),
    pd.DataFrame({'group': 'Urban', 'income': urban})
], ignore_index=True)
print("=== Sample Data ===")
print(df.head())
# --- 2. Descriptive Statistics ---
def desc(s):
    return pd.Series({
        'count': s.size,
        'mean': s.mean(),
        'median': s.median(),
        'mode': s.mode().iloc[0],
        'std': s.std(),
        'min': s.min(),
        'max': s.max(),
        'q1': s.quantile(0.25),
        'q3': s.quantile(0.75),
        'iqr': s.quantile(0.75)-s.quantile(0.25)
    })
summary = df.groupby('group')['income'].apply(desc)
print("\n=== Descriptive Summary by Group ===")
print(summary)
# --- 3. Visualizations ---
# Histogram with KDE
plt.figure(figsize=(8,5))
sns.histplot(df, x='income', hue='group', kde=True, bins=40, stat='density', common_norm=False)
plt.title('Histogram & Density: Rural vs Urban Income')
plt.xlabel('Income'); plt.ylabel('Density')
plt.tight_layout(); plt.savefig('hist_extended.png'); plt.close()
# Boxplot
plt.figure(figsize=(6,5))
sns.boxplot(df, x='group', y='income')
plt.title('Boxplot: Income by Group')
plt.ylabel('Income')
plt.tight_layout(); plt.savefig('box_extended.png'); plt.close()
# Violin Plot (shows distribution & density)
plt.figure(figsize=(6,5))
sns.violinplot(df, x='group', y='income')
plt.title('Violin Plot: Rural vs Urban')
plt.ylabel('Income')
plt.tight_layout(); plt.savefig('violin_extended.png'); plt.close()
# --- 4. Inequality Measure: Gini Coefficient ---
def gini(x):
    x = np.sort(x)
    n = len(x)
    return (2*np.sum(np.arange(1,n+1)*x)/(n*np.sum(x))) - (n+1)/n
print("\n=== Gini Coefficients ===")
for g in ['Rural','Urban']:
    gini_val = gini(df[df.group==g].income)
    print(f"{g}: {gini_val:.3f}")
# --- 5. Correlation & Regression ---
df['group_code'] = df['group'].map({'Rural':0,'Urban':1})
corr = df['income'].corr(df['group_code'])
print("\nCorrelation (group vs income):", round(corr,3))
# Simple OLS regression: income ~ group
X = sm.add_constant(df['group_code'])
model = sm.OLS(df['income'], X).fit()
print("\n=== Regression Summary ===")
print(model.summary())
# --- 6. Percentile Summary ---
percentiles = [10, 25, 50, 75, 90]
print("\n=== Percentile Summary ===")
for g in ['Rural','Urban']:
    vals = np.percentile(df[df.group==g].income, percentiles)
    print(f"{g}: {vals}")
# --- 7. Statistical Tests ---
# t-test (Welch’s)
t_stat, t_p = stats.ttest_ind(df[df.group=='Rural'].income, df[df.group=='Urban'].income, equal_var=False)
print("\nt-test (Welch) p-value:", round(t_p,5))
# Mann–Whitney U test
u_stat, u_p = stats.mannwhitneyu(df[df.group=='Rural'].income, df[df.group=='Urban'].income, alternative='two-sided')
print("Mann–Whitney U p-value:", round(u_p,5))
# --- 8. Extended Summary / Observations ---
print("\n=== Observations ===")
print("1. Urban incomes are higher on average (mean, median) and show a wider spread (higher std).")
print("2. Gini coefficient higher for Urban → more inequality in urban areas.")
print("3. Violin plot shows urban tail longer → some very high-income households.")
print("4. Regression: β1 ~ significant positive → urban effect on income is significant.")
print("5. t-test and Mann–Whitney reject H0 → income difference between Rural and Urban is statistically significant.")
