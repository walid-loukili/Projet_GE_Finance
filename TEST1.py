import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
# Assuming the data is in a CSV file named 'Financial_Statements.csv'
df = pd.read_csv('Financial_Statements.csv')

# Calculate missing metrics

# 1. Asset Turnover (Rotation des actifs)
# Calculate Total Assets based on Share Holder Equity and Debt/Equity Ratio
df['Total Assets'] = df['Share Holder Equity'] * (1 + df['Debt/Equity Ratio'])

# Calculate Asset Turnover (Rotation des actifs)
df['Asset Turnover'] = df['Revenue'] / df['Total Assets']

# 2. Total Debt (Dettes)
# Using Debt/Equity ratio: Debt = Share Holder Equity * Debt/Equity Ratio
df['Total Debt'] = df['Share Holder Equity'] * df['Debt/Equity Ratio']

# Additional metrics that might be useful:
# 3. Operating Margin (if needed)
df['Operating Margin'] = df['EBITDA'] / df['Revenue']

# 4. Price to Earnings (P/E) Ratio
df['P/E Ratio'] = df['Market Cap(in B USD)'] * 1000000000 / (df['Net Income'] * 1000000)  # Converting to same units

# 5. Price to Book Ratio
df['P/B Ratio'] = df['Market Cap(in B USD)'] * 1000000000 / (df['Share Holder Equity'] * 1000000)  # Converting to same units

# Save the updated dataset

print("Added metrics:")
print("- Asset Turnover (Rotation des actifs)")
print("- Total Debt (Dettes)")
print("- Operating Margin")
print("- P/E Ratio")
print("- P/B Ratio")
print("- Total Assets")

# Display the first few rows of the updated dataset with the new metrics
print("\nUpdated dataset preview:")
print(df[['Year', 'Company', 'Revenue', 'Total Assets', 'Asset Turnover', 'Total Debt', 'Operating Margin', 'P/E Ratio', 'P/B Ratio']].head())

# ------------- VISUALIZATION CODE ------------- #

# Set the style for the visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.figure(figsize=(14, 10))

# 1. Correlation heatmap between financial variables
# Select numeric columns for correlation analysis
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_metrics = ['ROA', 'ROE', 'Net Profit Margin', 'Debt/Equity Ratio', 
                       'Current Ratio', 'Asset Turnover', 'Operating Margin', 
                       'P/E Ratio', 'P/B Ratio']

# Use only columns that exist in the dataset
correlation_metrics = [col for col in correlation_metrics if col in df.columns]

plt.figure(figsize=(12, 10))
correlation_matrix = df[correlation_metrics].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Between Financial Variables', fontsize=16)
plt.tight_layout()
plt.savefig('financial_correlation_heatmap.png')
plt.close()

# 2. Performance comparison by sector
# Group data by sector and calculate mean values for key metrics
sector_performance = df.groupby('Category').agg({
    'ROA': 'mean',
    'ROE': 'mean', 
    'Net Profit Margin': 'mean',
    'Asset Turnover': 'mean',
    'Operating Margin': 'mean',
    'Current Ratio': 'mean',
    'Debt/Equity Ratio': 'mean'
}).reset_index()

# Function to create bar plots for sector comparison
def plot_sector_comparison(metric, title):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Category', y=metric, data=sector_performance)
    plt.title(f'{title} by Sector', fontsize=16)
    plt.xticks(rotation=45)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f'sector_comparison_{metric.replace("/", "_")}.png')
    plt.close()

# Create comparison plots for different metrics
plot_sector_comparison('ROE', 'Return on Equity')
plot_sector_comparison('ROA', 'Return on Assets')
plot_sector_comparison('Net Profit Margin', 'Net Profit Margin')
plot_sector_comparison('Asset Turnover', 'Asset Turnover')
plot_sector_comparison('Operating Margin', 'Operating Margin')

# 3. Multi-metric sector comparison (radar chart)
# Select top sectors for clarity
if len(sector_performance) > 6:
    top_sectors = sector_performance.nlargest(6, 'ROE')
else:
    top_sectors = sector_performance

# Prepare data for radar chart
metrics = ['ROA', 'ROE', 'Net Profit Margin', 'Asset Turnover', 'Operating Margin']
metrics = [m for m in metrics if m in sector_performance.columns]

# Normalize the data for radar chart
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
radar_data = top_sectors[metrics].copy()
radar_data[metrics] = scaler.fit_transform(radar_data[metrics])

# Create radar chart
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)

# Set the angles for each metric
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Close the loop

# Plot each sector
for i, sector in enumerate(top_sectors['Category']):
    values = radar_data.loc[radar_data.index[i], metrics].tolist()
    values += values[:1]  # Close the loop
    ax.plot(angles, values, linewidth=2, label=sector)
    ax.fill(angles, values, alpha=0.1)

# Set labels and title
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
plt.title('Multi-Metric Sector Comparison', size=16)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.tight_layout()
plt.savefig('sector_radar_comparison.png')
plt.close()

# 4. Time series analysis for selected companies in IT sector
it_companies = df[df['Category'] == 'IT'].copy()
if not it_companies.empty:
    for metric in ['ROE', 'ROA', 'Net Profit Margin']:
        plt.figure(figsize=(12, 6))
        for company in it_companies['Company'].unique():
            company_data = it_companies[it_companies['Company'] == company]
            plt.plot(company_data['Year'], company_data[metric], marker='o', label=company)
        
        plt.title(f'{metric} Trend for IT Companies', fontsize=16)
        plt.xlabel('Year')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'it_sector_{metric}_trend.png')
        plt.close()

print("\nVisualization complete! Generated the following files:")
print("1. financial_correlation_heatmap.png - Correlation between financial variables")
print("2. Multiple sector comparison bar charts (ROE, ROA, etc.)")
print("3. sector_radar_comparison.png - Multi-metric sector comparison")
print("4. Time series analysis charts for IT sector companies")