"""
Modèles de détection d'anomalies
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import os

# Configuration
DATA_PROCESSED = 'data/sp500_features.csv'
RESULTS_DIR = 'data/results/'

# Paramètres des modèles
ZSCORE_THRESHOLD = 3
IF_CONTAMINATION = 0.03
IF_ESTIMATORS = 100
LOF_NEIGHBORS = 20
LOF_CONTAMINATION = 0.03


def load_features():
    """
    Charge les données avec features
    """
    print("📂 Chargement des données avec features...")
    data = pd.read_csv(DATA_PROCESSED, index_col=0, parse_dates=True)
    print(f"   ✅ {len(data)} observations chargées")
    return data


def detect_zscore(data, threshold=ZSCORE_THRESHOLD):
    """
    Détection avec Z-Score (méthode statistique baseline)
    """
    print("\n" + "="*60)
    print("📊 MÉTHODE 1 : Z-SCORE (BASELINE)")
    print("="*60)
    
    print(f"\n🔍 Paramètres :")
    print(f"   Seuil : |Z-Score| > {threshold}")
    
    # Calculer le Z-Score
    mean_return = data['Return'].mean()
    std_return = data['Return'].std()
    
    data['ZScore'] = (data['Return'] - mean_return) / std_return
    
    print(f"\n📊 Statistiques Z-Score :")
    print(f"   Moyenne des rendements : {mean_return:.6f}")
    print(f"   Écart-type : {std_return:.6f}")
    print(f"   Z-Score min : {data['ZScore'].min():.2f}")
    print(f"   Z-Score max : {data['ZScore'].max():.2f}")
    
    # Détecter les anomalies
    data['ZScore_Anomaly'] = (np.abs(data['ZScore']) > threshold).astype(int)
    
    n_anomalies = data['ZScore_Anomaly'].sum()
    pct_anomalies = (n_anomalies / len(data)) * 100
    
    print(f"\n🎯 Résultats :")
    print(f"   Anomalies détectées : {n_anomalies}")
    print(f"   Pourcentage : {pct_anomalies:.2f}%")
    
    # Afficher les 10 plus grandes anomalies
    print(f"\n📉 Top 10 des anomalies (rendements les plus extrêmes) :")
    anomalies = data[data['ZScore_Anomaly'] == 1].copy()
    anomalies['Abs_ZScore'] = np.abs(anomalies['ZScore'])
    top_anomalies = anomalies.nlargest(10, 'Abs_ZScore')
    
    for idx, row in top_anomalies.iterrows():
        date = idx.strftime('%Y-%m-%d')
        return_pct = row['Return'] * 100
        zscore = row['ZScore']
        print(f"   {date} : Return={return_pct:+.2f}% | Z-Score={zscore:+.2f}")
    
    # Sauvegarder les résultats
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, 'zscore_anomalies.csv')
    anomalies[['Close', 'Return', 'ZScore', 'Volatility']].to_csv(result_path)
    print(f"\n💾 Résultats sauvegardés : {result_path}")
    
    print("\n" + "="*60)
    print("✅ MÉTHODE Z-SCORE TERMINÉE")
    print("="*60)
    
    return data


def detect_isolation_forest(data):
    """
    Détection avec Isolation Forest
    """
    print("\n" + "="*60)
    print("🤖 MÉTHODE 2 : ISOLATION FOREST")
    print("="*60)
    
    print(f"\n🔍 Paramètres :")
    print(f"   Contamination : {IF_CONTAMINATION} ({IF_CONTAMINATION*100}%)")
    print(f"   Nombre d'arbres : {IF_ESTIMATORS}")
    
    # Préparer les features
    features = ['Return', 'Volatility', 'Volume_Norm']
    X = data[features].values
    
    print(f"\n📊 Features utilisées : {features}")
    
    # Entraîner le modèle
    print(f"\n🔧 Entraînement du modèle...")
    model = IsolationForest(
        contamination=IF_CONTAMINATION,
        n_estimators=IF_ESTIMATORS,
        random_state=42
    )
    
    # Prédiction (-1 = anomalie, 1 = normal)
    predictions = model.fit_predict(X)
    
    # Convertir en 0/1 (0 = normal, 1 = anomalie)
    data['IF_Anomaly'] = (predictions == -1).astype(int)
    
    # Calculer le score d'anomalie
    data['IF_Score'] = model.score_samples(X)
    
    n_anomalies = data['IF_Anomaly'].sum()
    pct_anomalies = (n_anomalies / len(data)) * 100
    
    print(f"\n🎯 Résultats :")
    print(f"   Anomalies détectées : {n_anomalies}")
    print(f"   Pourcentage : {pct_anomalies:.2f}%")
    
    # Afficher les 10 plus grandes anomalies
    print(f"\n📉 Top 10 des anomalies (scores les plus bas) :")
    anomalies = data[data['IF_Anomaly'] == 1].copy()
    top_anomalies = anomalies.nsmallest(10, 'IF_Score')
    
    for idx, row in top_anomalies.iterrows():
        date = idx.strftime('%Y-%m-%d')
        return_pct = row['Return'] * 100
        score = row['IF_Score']
        print(f"   {date} : Return={return_pct:+.2f}% | Score={score:.4f}")
    
    # Sauvegarder les résultats
    result_path = os.path.join(RESULTS_DIR, 'isolation_forest_anomalies.csv')
    anomalies[['Close', 'Return', 'IF_Score', 'Volatility']].to_csv(result_path)
    print(f"\n💾 Résultats sauvegardés : {result_path}")
    
    print("\n" + "="*60)
    print("✅ MÉTHODE ISOLATION FOREST TERMINÉE")
    print("="*60)
    
    return data


def detect_lof(data):
    """
    Détection avec Local Outlier Factor
    """
    print("\n" + "="*60)
    print("🤖 MÉTHODE 3 : LOCAL OUTLIER FACTOR (LOF)")
    print("="*60)
    
    print(f"\n🔍 Paramètres :")
    print(f"   Nombre de voisins : {LOF_NEIGHBORS}")
    print(f"   Contamination : {LOF_CONTAMINATION} ({LOF_CONTAMINATION*100}%)")
    
    # Préparer les features
    features = ['Return', 'Volatility', 'Volume_Norm']
    X = data[features].values
    
    print(f"\n📊 Features utilisées : {features}")
    
    # Entraîner le modèle
    print(f"\n🔧 Entraînement du modèle...")
    model = LocalOutlierFactor(
        n_neighbors=LOF_NEIGHBORS,
        contamination=LOF_CONTAMINATION
    )
    
    # Prédiction (-1 = anomalie, 1 = normal)
    predictions = model.fit_predict(X)
    
    # Convertir en 0/1
    data['LOF_Anomaly'] = (predictions == -1).astype(int)
    
    # Score LOF (valeurs négatives = anomalies)
    data['LOF_Score'] = model.negative_outlier_factor_
    
    n_anomalies = data['LOF_Anomaly'].sum()
    pct_anomalies = (n_anomalies / len(data)) * 100
    
    print(f"\n🎯 Résultats :")
    print(f"   Anomalies détectées : {n_anomalies}")
    print(f"   Pourcentage : {pct_anomalies:.2f}%")
    
    # Afficher les 10 plus grandes anomalies
    print(f"\n📉 Top 10 des anomalies (scores les plus bas) :")
    anomalies = data[data['LOF_Anomaly'] == 1].copy()
    top_anomalies = anomalies.nsmallest(10, 'LOF_Score')
    
    for idx, row in top_anomalies.iterrows():
        date = idx.strftime('%Y-%m-%d')
        return_pct = row['Return'] * 100
        score = row['LOF_Score']
        print(f"   {date} : Return={return_pct:+.2f}% | Score={score:.4f}")
    
    # Sauvegarder les résultats
    result_path = os.path.join(RESULTS_DIR, 'lof_anomalies.csv')
    anomalies[['Close', 'Return', 'LOF_Score', 'Volatility']].to_csv(result_path)
    print(f"\n💾 Résultats sauvegardés : {result_path}")
    
    print("\n" + "="*60)
    print("✅ MÉTHODE LOF TERMINÉE")
    print("="*60)
    
    return data


def run_all_models():
    """
    Exécute les 3 modèles
    """
    print("\n" + "="*60)
    print("🚀 DÉTECTION DES ÉVÉNEMENTS EXTRÊMES")
    print("="*60)
    
    # Charger les données
    data = load_features()
    
    # Modèle 1 : Z-Score
    data = detect_zscore(data)
    
    # Modèle 2 : Isolation Forest
    data = detect_isolation_forest(data)
    
    # Modèle 3 : LOF
    data = detect_lof(data)
    
    # Sauvegarder le dataset complet
    final_path = os.path.join(RESULTS_DIR, 'all_detections.csv')
    data.to_csv(final_path)
    print(f"\n💾 Dataset complet sauvegardé : {final_path}")
    
    print("\n" + "="*60)
    print("🎉 TOUTES LES DÉTECTIONS TERMINÉES")
    print("="*60)
    
    return data
if __name__ == "__main__":
    df = run_all_models()
    print("\n✅ Module models.py fonctionne correctement !")