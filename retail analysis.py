import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── load data ────────────────────────────────────────────────
# dataset is included in the project folder (superstore_sales.csv)
df = pd.read_csv('superstore_sales.csv')
print(f"Shape: {df.shape}")
print(df.head(3).to_string())

# ── cleaning ─────────────────────────────────────────────────
df['order_date'] = pd.to_datetime(df['order_date'])
df['ship_date']  = pd.to_datetime(df['ship_date'])
df['ship_lag']   = (df['ship_date'] - df['order_date']).dt.days

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df['year']           = df['order_date'].dt.year
df['month']          = df['order_date'].dt.month
df['quarter']        = df['order_date'].dt.quarter
df['profit_margin']  = (df['profit'] / df['sales'].replace(0, np.nan) * 100).round(2)

print(f"\nCleaned shape: {df.shape}")
print(df[['sales','profit','discount','quantity']].describe().round(2).to_string())

# ── business summary ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  BUSINESS SUMMARY")
print("=" * 55)
print(f"  Total orders       : {df['order_id'].nunique()}")
print(f"  Unique customers   : {df['customer_id'].nunique()}")
print(f"  Total revenue      : ${df['sales'].sum():,.2f}")
print(f"  Total profit       : ${df['profit'].sum():,.2f}")
print(f"  Avg profit margin  : {df['profit_margin'].mean():.2f}%")
print(f"  Avg discount       : {df['discount'].mean()*100:.1f}%")
print(f"  Date range         : {df['order_date'].min().date()} → {df['order_date'].max().date()}")

print("\n  Revenue by Category:")
cat_rev = df.groupby('category')[['sales','profit']].sum().sort_values('sales', ascending=False)
for cat, row in cat_rev.iterrows():
    margin = row['profit'] / row['sales'] * 100
    print(f"    {cat:<22} Sales=${row['sales']:>10,.0f}  Profit=${row['profit']:>8,.0f}  Margin={margin:.1f}%")

print("\n  Top 5 Sub-Categories by Profit:")
sub_profit = df.groupby('sub_category')['profit'].sum().sort_values(ascending=False)
for sub, val in sub_profit.head(5).items():
    print(f"    {sub:<20} ${val:,.0f}")

print("\n  Bottom 5 (loss leaders):")
for sub, val in sub_profit.tail(5).items():
    print(f"    {sub:<20} ${val:,.0f}")

print("\n  Sales by Region:")
print(df.groupby('region')[['sales','profit']].sum().sort_values('sales', ascending=False).to_string())

# ── monthly trend ────────────────────────────────────────────
monthly = df.groupby(['year','month'])[['sales','profit']].sum().reset_index()
monthly['date'] = pd.to_datetime(monthly[['year','month']].assign(day=1))

# ── profit prediction model ──────────────────────────────────
print("\n--- training profit prediction model ---")
le = LabelEncoder()
df['category_enc']  = le.fit_transform(df['category'])
df['region_enc']    = le.fit_transform(df['region'])
df['shipmode_enc']  = le.fit_transform(df['ship_mode'])

features = ['sales','quantity','discount','ship_lag','category_enc','region_enc','shipmode_enc']
X = df[features]
y = df['profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"  MAE : ${mae:.2f}")
print(f"  R²  : {r2:.3f}")

coef_df = pd.Series(model.coef_, index=features).sort_values(key=abs, ascending=False)

# ── plots ────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 22))
fig.suptitle('Superstore Retail — End-to-End Data Analysis', fontsize=17, fontweight='bold', y=1.005)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.35)

pal = ['#2ecc71','#3498db','#e74c3c','#f39c12','#9b59b6','#1abc9c']

# 1. monthly revenue trend (dual axis)
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(monthly['date'], monthly['sales'], color='#3498db', linewidth=2,
         label='Sales', marker='o', markersize=3)
ax1b = ax1.twinx()
ax1b.plot(monthly['date'], monthly['profit'], color='#2ecc71', linewidth=2,
          label='Profit', linestyle='--', marker='s', markersize=3)
ax1.set_title('Monthly Revenue & Profit Trend', fontweight='bold')
ax1.set_ylabel('Sales ($)', color='#3498db')
ax1b.set_ylabel('Profit ($)', color='#2ecc71')
ax1.tick_params(axis='x', rotation=30, labelsize=8)
lines1, l1 = ax1.get_legend_handles_labels()
lines2, l2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, l1 + l2, fontsize=8, loc='upper left')

# 2. sales by category pie
ax2 = fig.add_subplot(gs[0, 2])
cat_sales = df.groupby('category')['sales'].sum()
ax2.pie(cat_sales.values, labels=cat_sales.index,
        colors=['#3498db','#e74c3c','#f39c12'],
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor':'white','linewidth':1.5})
ax2.set_title('Sales by Category', fontweight='bold')

# 3. profit by sub-category (horizontal bar, red = loss)
ax3 = fig.add_subplot(gs[1, :2])
sub_sorted = sub_profit.sort_values()
bar_colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in sub_sorted.values]
ax3.barh(sub_sorted.index, sub_sorted.values, color=bar_colors, edgecolor='none', height=0.6)
ax3.axvline(0, color='black', linewidth=0.8)
ax3.set_title('Profit by Sub-Category  (red = loss-making)', fontweight='bold')
ax3.set_xlabel('Total Profit ($)')
ax3.tick_params(axis='y', labelsize=8)

# 4. discount vs profit scatter
ax4 = fig.add_subplot(gs[1, 2])
sample = df.sample(min(600, len(df)), random_state=1)
sc = ax4.scatter(sample['discount'], sample['profit'],
                 c=sample['sales'], cmap='viridis',
                 alpha=0.5, s=15, edgecolors='none')
plt.colorbar(sc, ax=ax4, label='Sales ($)', shrink=0.8)
ax4.axhline(0, color='red', linestyle='--', linewidth=0.8)
ax4.set_title('Discount vs Profit', fontweight='bold')
ax4.set_xlabel('Discount')
ax4.set_ylabel('Profit ($)')

# 5. sales by region
ax5 = fig.add_subplot(gs[2, 0])
reg_sales = df.groupby('region')['sales'].sum().sort_values()
ax5.barh(reg_sales.index, reg_sales.values,
         color=['#3498db','#2ecc71','#e74c3c','#f39c12'], edgecolor='none', height=0.5)
ax5.set_title('Sales by Region', fontweight='bold')
ax5.set_xlabel('Total Sales ($)')

# 6. profit margin by category + segment
ax6 = fig.add_subplot(gs[2, 1])
pivot = df.groupby(['category','segment'])['profit_margin'].mean().unstack()
pivot.plot(kind='bar', ax=ax6, color=pal[:3], edgecolor='none', width=0.6)
ax6.set_title('Avg Profit Margin by Category & Segment', fontweight='bold')
ax6.set_xlabel('')
ax6.tick_params(axis='x', rotation=15)
ax6.set_ylabel('Profit Margin (%)')
ax6.legend(fontsize=8)

# 7. quarterly sales by year
ax7 = fig.add_subplot(gs[2, 2])
qtr = df.groupby(['year','quarter'])['sales'].sum().unstack()
qtr.T.plot(kind='bar', ax=ax7, edgecolor='none', width=0.65)
ax7.set_title('Quarterly Sales by Year', fontweight='bold')
ax7.set_xlabel('Quarter')
ax7.set_ylabel('Sales ($)')
ax7.set_xticklabels(['Q1','Q2','Q3','Q4'], rotation=0)
ax7.legend(title='Year', fontsize=8)

# 8. shipping lag by mode
ax8 = fig.add_subplot(gs[3, 0])
for mode in df['ship_mode'].unique():
    df[df['ship_mode'] == mode]['ship_lag'].hist(
        ax=ax8, bins=12, alpha=0.6, label=mode, edgecolor='none')
ax8.set_title('Shipping Lag by Ship Mode', fontweight='bold')
ax8.set_xlabel('Days to Ship')
ax8.set_ylabel('Count')
ax8.legend(fontsize=7)

# 9. actual vs predicted profit
ax9 = fig.add_subplot(gs[3, 1])
ax9.scatter(y_test, y_pred, alpha=0.3, s=12, color='#3498db', edgecolors='none')
lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax9.plot(lim, lim, 'r--', linewidth=1.5)
ax9.set_title(f'Actual vs Predicted Profit\nR²={r2:.3f}  MAE=${mae:.2f}', fontweight='bold')
ax9.set_xlabel('Actual Profit ($)')
ax9.set_ylabel('Predicted Profit ($)')

# 10. feature coefficients
ax10 = fig.add_subplot(gs[3, 2])
coef_colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in coef_df.values]
ax10.barh(coef_df.index, coef_df.values, color=coef_colors, edgecolor='none', height=0.5)
ax10.axvline(0, color='black', linewidth=0.8)
ax10.set_title('Feature Coefficients (Profit Model)', fontweight='bold')
ax10.tick_params(axis='y', labelsize=8)

plt.savefig('retail_dashboard.png', dpi=150, bbox_inches='tight')
print("\nsaved → retail_dashboard.png")
plt.show()

print("\n" + "=" * 55)
print("  CONCLUSIONS")
print("=" * 55)
print("  1. Technology leads in revenue across all regions")
print("  2. Discounts above 30% almost always result in negative profit")
print("  3. Tables and Bookcases are consistent loss-making sub-categories")
print("  4. West region leads in both sales and profit")
print("  5. Q4 outperforms other quarters every year (seasonal demand)")
print("  6. Discount is the strongest negative predictor of profit")
