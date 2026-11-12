"""
Téléchargement des données S&P 500 depuis Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import os

# Configuration directe
TICKER = '^GSPC'
START_DATE = '2010-01-01'
END_DATE = '2024-12-31'
DATA_RAW = 'data/sp500_raw.csv'

def download_sp500():
    """
    Télécharge les données historiques du S&P 500
    """
    print("="*60)
    print("📥 ÉTAPE 1 : TÉLÉCHARGEMENT DES DONNÉES")
    print("="*60)
    
    print(f"\n🔍 Paramètres :")
    print(f"   Indice : {TICKER} (S&P 500)")
    print(f"   Période : {START_DATE} à {END_DATE}")
    
    # Téléchargement
    print(f"\n📊 Téléchargement en cours...")
    data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    
    if data.empty:
        print("❌ Erreur : Aucune donnée téléchargée")
        return None
    
    # Simplifier les colonnes (enlever le ticker)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Informations sur les données
    print(f"\n✅ Téléchargement réussi !")
    print(f"   Nombre de jours : {len(data)}")
    print(f"   Première date : {data.index[0].strftime('%Y-%m-%d')}")
    print(f"   Dernière date : {data.index[-1].strftime('%Y-%m-%d')}")
    
    # Afficher les colonnes
    print(f"\n📋 Colonnes disponibles :")
    for col in data.columns:
        print(f"   - {col}")
    
    # Vérifier les valeurs manquantes
    missing = data.isnull().sum()
    print(f"\n🔍 Valeurs manquantes :")
    if missing.sum() == 0:
        print("   ✅ Aucune valeur manquante")
    else:
        for col in missing.index:
            if missing[col] > 0:
                print(f"   ⚠️ {col}: {missing[col]} valeurs manquantes")
    
    # Afficher aperçu
    print(f"\n📈 Aperçu des 5 premières lignes :")
    print(data.head())
    
    print(f"\n📉 Aperçu des 5 dernières lignes :")
    print(data.tail())
    
    # Statistiques descriptives
    print(f"\n📊 Statistiques du prix de clôture :")
    print(f"   Min : ${data['Close'].min():.2f}")
    print(f"   Max : ${data['Close'].max():.2f}")
    print(f"   Moyenne : ${data['Close'].mean():.2f}")
    print(f"   Médiane : ${data['Close'].median():.2f}")
    
    # Sauvegarder
    os.makedirs(os.path.dirname(DATA_RAW), exist_ok=True)
    data.to_csv(DATA_RAW)
    print(f"\n💾 Données sauvegardées : {DATA_RAW}")
    
    print("\n" + "="*60)
    print("✅ ÉTAPE 1 TERMINÉE")
    print("="*60)
    
    return data

if __name__ == "__main__":
    df = download_sp500()
    if df is not None:
        print(f"\n🎉 Module data_loader.py fonctionne correctement !")