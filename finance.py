# Analyse Financière Enrichie par des Techniques de Data Science et Machine Learning
# Travail à Faire en Groupe de 4

# 0. Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
import warnings
warnings.filterwarnings('ignore')
import matplotlib.style as style
style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 1. Chargement des données
# Dans un environnement réel, vous téléchargeriez le dataset depuis Kaggle
# Pour cet exemple, nous allons supposer que vous avez déjà téléchargé le fichier

print("1. CHARGEMENT ET EXPLORATION DES DONNÉES")
print("-" * 80)

# Remplacez ce chemin par le vôtre
try:
    # Tentative de charger directement avec l'URL de Kaggle si vous avez configuré l'API
    df = pd.read_csv('https://www.kaggle.com/datasets/atharvaarya25/financials/download')
except:
    print("Impossible de charger directement depuis Kaggle. Veuillez télécharger manuellement le dataset.")
    print("Pour cet exemple, nous allons créer un échantillon de données simulé.")
    
    # Création d'un dataset synthétique pour l'exemple
    np.random.seed(42)
    sectors = ['Technology', 'Finance', 'Healthcare', 'Consumer Goods', 'Energy', 'Utilities', 'Real Estate']
    companies = [f"Company_{i}" for i in range(1, 101)]
    
    df = pd.DataFrame({
        'Company Name': np.random.choice(companies, 100),
        'Ticker': [f"TKR{i}" for i in range(1, 101)],
        'Sector': np.random.choice(sectors, 100),
        'Total Revenue': np.random.uniform(100, 10000, 100) * 1e6,
        'Gross Profit': np.random.uniform(50, 5000, 100) * 1e6,
        'Operating Income': np.random.uniform(20, 3000, 100) * 1e6,
        'Net Income': np.random.uniform(-500, 2500, 100) * 1e6,
        'Total Assets': np.random.uniform(500, 20000, 100) * 1e6,
        'Total Liabilities': np.random.uniform(200, 15000, 100) * 1e6,
        'Equity': np.random.uniform(100, 10000, 100) * 1e6,
        'Cash and Cash Equivalents': np.random.uniform(50, 2000, 100) * 1e6,
        'Earnings Per Share (EPS)': np.random.uniform(-5, 25, 100),
        'Price-to-Earnings Ratio (P/E)': np.random.uniform(5, 50, 100),
        'Dividend Yield': np.random.uniform(0, 0.08, 100),
        'Market Capitalization': np.random.uniform(1000, 50000, 100) * 1e6
    })
    
    # Ajustement pour assurer la cohérence des données financières
    df['Total Liabilities'] = np.minimum(df['Total Liabilities'], df['Total Assets'] * 0.9)
    df['Equity'] = df['Total Assets'] - df['Total Liabilities']
    df['Gross Profit'] = np.minimum(df['Gross Profit'], df['Total Revenue'] * 0.8)
    df['Operating Income'] = np.minimum(df['Operating Income'], df['Gross Profit'] * 0.9)
    df['Net Income'] = np.minimum(df['Net Income'], df['Operating Income'] * 1.2)
    df['Cash and Cash Equivalents'] = np.minimum(df['Cash and Cash Equivalents'], df['Total Assets'] * 0.4)

# Aperçu des données
print("\nAperçu des premières lignes:")
print(df.head())

print("\nInformations sur le dataset:")
print(df.info())

print("\nStatistiques descriptives:")
print(df.describe())

print("\nVérification des valeurs manquantes:")
print(df.isnull().sum())

# 2. PRÉTRAITEMENT DES DONNÉES
print("\n\n2. PRÉTRAITEMENT DES DONNÉES")
print("-" * 80)

# Gestion des valeurs manquantes
if df.isnull().sum().sum() > 0:
    print("Gestion des valeurs manquantes...")
    df = df.fillna(df.median())  # On remplace par la médiane pour les valeurs numériques

# Détection des valeurs aberrantes
print("\nDétection des valeurs aberrantes avec la méthode IQR...")
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    if outliers > 0:
        print(f"  - {col}: {outliers} valeurs aberrantes détectées")
        # Nous ne supprimons pas les outliers tout de suite, car ils peuvent être significatifs en finance

# 3. CALCUL DES RATIOS FINANCIERS
print("\n\n3. CALCUL DES RATIOS FINANCIERS")
print("-" * 80)

print("Calcul des ratios financiers clés...")

# Rentabilité
df['ROE'] = df['Net Income'] / df['Equity']  # Return on Equity
df['ROA'] = df['Net Income'] / df['Total Assets']  # Return on Assets
df['Net Margin'] = df['Net Income'] / df['Total Revenue']  # Marge nette
df['Gross Margin'] = df['Gross Profit'] / df['Total Revenue']  # Marge brute
df['Operating Margin'] = df['Operating Income'] / df['Total Revenue']  # Marge opérationnelle

# Endettement
df['Debt to Equity'] = df['Total Liabilities'] / df['Equity']  # Ratio d'endettement
df['Debt to Assets'] = df['Total Liabilities'] / df['Total Assets']  # Ratio de dette sur actifs

# Liquidité
df['Current Ratio'] = df['Cash and Cash Equivalents'] / (df['Total Liabilities'] * 0.3)  # Ratio approximatif de liquidité

# Efficacité
df['Asset Turnover'] = df['Total Revenue'] / df['Total Assets']  # Rotation des actifs

# Création d'un indicateur de performance globale
df['Performance Score'] = (
    df['ROE'].rank(pct=True) + 
    df['ROA'].rank(pct=True) + 
    df['Net Margin'].rank(pct=True) - 
    df['Debt to Equity'].rank(pct=True)
) / 4

# Affichage des nouveaux ratios
print("\nAperçu des ratios calculés:")
ratios_cols = ['ROE', 'ROA', 'Net Margin', 'Gross Margin', 'Operating Margin', 
              'Debt to Equity', 'Debt to Assets', 'Current Ratio', 'Asset Turnover', 'Performance Score']
print(df[ratios_cols].describe())

# 4. EXPLORATION DES DONNÉES (EDA - Analyse descriptive)
print("\n\n4. EXPLORATION DES DONNÉES (EDA)")
print("-" * 80)

print("Analyse de la distribution des principales variables financières...")

# Distributions des variables principales
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

# Liste des variables à visualiser
vars_to_plot = ['Total Revenue', 'Net Income', 'Total Assets', 'ROE', 'ROA', 
                'Net Margin', 'Debt to Equity', 'Asset Turnover', 'Performance Score']

for i, var in enumerate(vars_to_plot):
    sns.histplot(df[var], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution de {var}')
    
plt.tight_layout()
plt.savefig('distributions_variables.png')
plt.close()
print("  - Figure 'distributions_variables.png' sauvegardée.")

# Matrice de corrélation
plt.figure(figsize=(16, 14))
correlation_vars = ['Total Revenue', 'Net Income', 'Total Assets', 'Equity', 'ROE', 'ROA', 
                    'Net Margin', 'Debt to Equity', 'Asset Turnover', 'Performance Score']
correlation_matrix = df[correlation_vars].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Matrice de corrélation des variables financières')
plt.savefig('correlation_matrix.png')
plt.close()
print("  - Figure 'correlation_matrix.png' sauvegardée.")

# Performance par secteur
plt.figure(figsize=(14, 8))
sector_perf = df.groupby('Sector')['Performance Score'].mean().sort_values(ascending=False)
sector_perf.plot(kind='bar', color='teal')
plt.title('Performance moyenne par secteur')
plt.ylabel('Score de performance')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('sector_performance.png')
plt.close()
print("  - Figure 'sector_performance.png' sauvegardée.")

# 5. VISUALISATION DE CORRÉLATION ENTRE VARIABLES FINANCIÈRES
print("\n\n5. VISUALISATION DE CORRÉLATION ENTRE VARIABLES FINANCIÈRES")
print("-" * 80)

print("Création de visualisations pour les relations entre variables...")

# Corrélation entre ROE et ROA avec coloration par secteur
plt.figure(figsize=(12, 9))
sns.scatterplot(data=df, x='ROA', y='ROE', hue='Sector', size='Total Revenue', 
                sizes=(50, 300), alpha=0.7)
plt.title('Relation entre ROA et ROE par secteur')
plt.xlabel('Return on Assets (ROA)')
plt.ylabel('Return on Equity (ROE)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('roe_roa_scatter.png')
plt.close()
print("  - Figure 'roe_roa_scatter.png' sauvegardée.")

# Relation entre revenu total et marge nette
plt.figure(figsize=(12, 9))
sns.scatterplot(data=df, x='Total Revenue', y='Net Margin', hue='Sector', 
                size='Total Assets', sizes=(50, 300), alpha=0.7)
plt.title('Relation entre Revenu Total et Marge Nette')
plt.xlabel('Revenu Total')
plt.ylabel('Marge Nette')
plt.xscale('log')  # Échelle logarithmique pour mieux visualiser
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('revenue_margin_scatter.png')
plt.close()
print("  - Figure 'revenue_margin_scatter.png' sauvegardée.")

# Pairplot des ratios financiers clés
pairplot_vars = ['ROE', 'ROA', 'Net Margin', 'Debt to Equity']
sns.pairplot(df, vars=pairplot_vars, hue='Sector', height=2.5, corner=True)
plt.savefig('ratios_pairplot.png')
plt.close()
print("  - Figure 'ratios_pairplot.png' sauvegardée.")
