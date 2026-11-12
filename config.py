"""
Configuration du projet
"""

# ===== DONNÉES =====
TICKER = '^GSPC'
START_DATE = '2010-01-01'
END_DATE = '2024-12-31'

# Chemins
DATA_RAW = 'data/sp500_raw.csv'
DATA_PROCESSED = 'data/sp500_features.csv'
RESULTS_PATH = 'data/results.csv'

# ===== FEATURES =====
VOLATILITY_WINDOW = 20  # Fenêtre pour volatilité historique

# ===== MODÈLES =====

# Z-Score
ZSCORE_THRESHOLD = 3  # |z| > 3 = anomalie

# Isolation Forest
IF_CONTAMINATION = 0.03  # 3% d'anomalies attendues
IF_ESTIMATORS = 100
IF_RANDOM_STATE = 42

# Local Outlier Factor
LOF_NEIGHBORS = 20
LOF_CONTAMINATION = 0.03

# ===== VISUALISATION =====
FIGURE_SIZE = (14, 8)
DPI = 100