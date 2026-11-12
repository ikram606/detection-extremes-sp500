"""
Prétraitement et Feature Engineering
"""

import pandas as pd
import numpy as np
import os

# Configuration
DATA_RAW = 'data/sp500_raw.csv'
DATA_PROCESSED = 'data/sp500_features.csv'
VOLATILITY_WINDOW = 20

def load_raw_data():
    """
    Charge les données brutes
    """
    print("📂 Chargement des données brutes...")
    data = pd.read_csv(DATA_RAW, index_col=0, parse_dates=True)
    print(f"   ✅ {len(data)} lignes chargées")
    return data

def clean_data(data):
    """
    Nettoyage des données
    """
    print("\n🧹 Nettoyage des données...")
    
    # Trier par date
    data = data.sort_index()
    print("   ✅ Données triées par date")
    
    # Vérifier les doublons
    duplicates = data.index.duplicated().sum()
    if duplicates > 0:
        print(f"   ⚠️ {duplicates} doublons supprimés")
        data = data[~data.index.duplicated(keep='first')]
    else:
        print("   ✅ Aucun doublon")
    
    # Vérifier les valeurs manquantes
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print(f"   ⚠️ Valeurs manquantes détectées :")
        for col in missing.index:
            if missing[col] > 0:
                print(f"      - {col}: {missing[col]}")
        # Supprimer les lignes avec valeurs manquantes
        data = data.dropna()
        print(f"   ✅ Lignes avec NA supprimées. Reste : {len(data)} lignes")
    else:
        print("   ✅ Aucune valeur manquante")
    
    return data

def calculate_returns(data):
    """
    Calcule les rendements logarithmiques
    """
    print("\n📊 Calcul des rendements logarithmiques...")
    
    # Rendement = ln(P_t / P_{t-1})
    data['Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Supprimer la première ligne (NaN)
    data = data.dropna(subset=['Return'])
    
    print(f"   ✅ Rendements calculés")
    print(f"   Rendement moyen : {data['Return'].mean():.6f}")
    print(f"   Écart-type : {data['Return'].std():.6f}")
    print(f"   Min : {data['Return'].min():.6f}")
    print(f"   Max : {data['Return'].max():.6f}")
    
    return data

def calculate_volatility(data):
    """
    Calcule la volatilité historique (fenêtre glissante)
    """
    print(f"\n📈 Calcul de la volatilité (fenêtre {VOLATILITY_WINDOW} jours)...")
    
    # Volatilité = écart-type des rendements sur fenêtre glissante
    data['Volatility'] = data['Return'].rolling(window=VOLATILITY_WINDOW).std()
    
    # Supprimer les premières lignes avec NaN
    data = data.dropna(subset=['Volatility'])
    
    print(f"   ✅ Volatilité calculée")
    print(f"   Volatilité moyenne : {data['Volatility'].mean():.6f}")
    print(f"   Volatilité min : {data['Volatility'].min():.6f}")
    print(f"   Volatilité max : {data['Volatility'].max():.6f}")
    
    return data

def normalize_volume(data):
    """
    Normalise le volume (z-score)
    """
    print(f"\n📊 Normalisation du volume...")
    
    # Volume normalisé = (Volume - mean) / std
    data['Volume_Norm'] = (data['Volume'] - data['Volume'].mean()) / data['Volume'].std()
    
    print(f"   ✅ Volume normalisé")
    
    return data

def prepare_features():
    """
    Pipeline complet de prétraitement
    """
    print("="*60)
    print("🔧 ÉTAPE 2 : PRÉTRAITEMENT & FEATURE ENGINEERING")
    print("="*60)
    
    # 1. Charger
    data = load_raw_data()
    
    # 2. Nettoyer
    data = clean_data(data)
    
    # 3. Calculer rendements
    data = calculate_returns(data)
    
    # 4. Calculer volatilité
    data = calculate_volatility(data)
    
    # 5. Normaliser volume
    data = normalize_volume(data)
    
    # Résumé final
    print("\n" + "="*60)
    print("📋 RÉSUMÉ DES FEATURES")
    print("="*60)
    print(f"Nombre d'observations : {len(data)}")
    print(f"\nColonnes disponibles :")
    for col in data.columns:
        print(f"   - {col}")
    
    print(f"\n📈 Aperçu des données finales :")
    print(data[['Close', 'Return', 'Volatility', 'Volume_Norm']].head(10))
    
    print(f"\n📊 Statistiques des features :")
    print(data[['Return', 'Volatility', 'Volume_Norm']].describe())
    
    # Sauvegarder
    os.makedirs(os.path.dirname(DATA_PROCESSED), exist_ok=True)
    data.to_csv(DATA_PROCESSED)
    print(f"\n💾 Données sauvegardées : {DATA_PROCESSED}")
    
    print("\n" + "="*60)
    print("✅ ÉTAPE 2 TERMINÉE")
    print("="*60)
    
    return data

if __name__ == "__main__":
    df = prepare_features()
    if df is not None:
        print(f"\n🎉 Module preprocessing.py fonctionne correctement !")