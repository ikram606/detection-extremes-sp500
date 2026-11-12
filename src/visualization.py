"""
Visualisation et comparaison des résultats
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configuration
DATA_RESULTS = 'data/results/all_detections.csv'
OUTPUT_DIR = 'outputs/figures/'

# Style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def load_results():
    """
    Charge les résultats complets
    """
    print("📂 Chargement des résultats...")
    data = pd.read_csv(DATA_RESULTS, index_col=0, parse_dates=True)
    print(f"   ✅ {len(data)} observations chargées")
    return data


def plot_price_evolution(data):
    """
    Graphique : Évolution du prix S&P 500
    """
    print("\n📈 Création du graphique : Évolution du prix...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(data.index, data['Close'], color='steelblue', linewidth=1.5)
    ax.set_title('Évolution du S&P 500 (2010-2024)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Prix de clôture ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, '01_price_evolution.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Sauvegardé : {path}")
    plt.close()


def plot_returns_distribution(data):
    """
    Graphique : Distribution des rendements
    """
    print("\n📊 Création du graphique : Distribution des rendements...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogramme
    ax1.hist(data['Return'], bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(data['Return'].mean(), color='red', linestyle='--', linewidth=2, label='Moyenne')
    ax1.axvline(data['Return'].mean() + 3*data['Return'].std(), color='orange', linestyle='--', linewidth=2, label='±3σ')
    ax1.axvline(data['Return'].mean() - 3*data['Return'].std(), color='orange', linestyle='--', linewidth=2)
    ax1.set_title('Distribution des rendements journaliers', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Rendement', fontsize=11)
    ax1.set_ylabel('Fréquence', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(data['Return'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_title('Box plot des rendements', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Rendement', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, '02_returns_distribution.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Sauvegardé : {path}")
    plt.close()


def plot_anomalies_zscore(data):
    """
    Graphique : Anomalies détectées par Z-Score
    """
    print("\n📊 Création du graphique : Z-Score anomalies...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Prix avec anomalies
    ax1.plot(data.index, data['Close'], color='steelblue', linewidth=1, label='S&P 500')
    
    anomalies = data[data['ZScore_Anomaly'] == 1]
    ax1.scatter(anomalies.index, anomalies['Close'], color='red', s=50, 
                marker='v', label=f'Anomalies Z-Score (n={len(anomalies)})', zorder=5)
    
    ax1.set_title('Détection Z-Score : Événements extrêmes sur le S&P 500', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Prix de clôture ($)', fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Rendements avec seuils
    ax2.plot(data.index, data['Return']*100, color='gray', linewidth=0.8, alpha=0.6)
    ax2.scatter(anomalies.index, anomalies['Return']*100, color='red', s=50, 
                marker='v', label='Anomalies', zorder=5)
    ax2.axhline(3*data['Return'].std()*100, color='orange', linestyle='--', 
                linewidth=2, label='Seuil ±3σ')
    ax2.axhline(-3*data['Return'].std()*100, color='orange', linestyle='--', linewidth=2)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    ax2.set_title('Rendements journaliers et seuils de détection', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Rendement (%)', fontsize=11)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, '03_zscore_detection.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Sauvegardé : {path}")
    plt.close()


def plot_anomalies_if(data):
    """
    Graphique : Anomalies détectées par Isolation Forest
    """
    print("\n🤖 Création du graphique : Isolation Forest anomalies...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(data.index, data['Close'], color='steelblue', linewidth=1, label='S&P 500')
    
    anomalies = data[data['IF_Anomaly'] == 1]
    ax.scatter(anomalies.index, anomalies['Close'], color='purple', s=50, 
               marker='s', label=f'Anomalies IF (n={len(anomalies)})', zorder=5)
    
    ax.set_title('Détection Isolation Forest : Événements extrêmes', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Prix de clôture ($)', fontsize=11)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, '04_isolation_forest_detection.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Sauvegardé : {path}")
    plt.close()


def plot_anomalies_lof(data):
    """
    Graphique : Anomalies détectées par LOF
    """
    print("\n🤖 Création du graphique : LOF anomalies...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(data.index, data['Close'], color='steelblue', linewidth=1, label='S&P 500')
    
    anomalies = data[data['LOF_Anomaly'] == 1]
    ax.scatter(anomalies.index, anomalies['Close'], color='green', s=50, 
               marker='D', label=f'Anomalies LOF (n={len(anomalies)})', zorder=5)
    
    ax.set_title('Détection LOF : Événements extrêmes', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Prix de clôture ($)', fontsize=11)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, '05_lof_detection.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Sauvegardé : {path}")
    plt.close()


def plot_comparison(data):
    """
    Graphique : Comparaison des 3 méthodes
    """
    print("\n🔍 Création du graphique : Comparaison des méthodes...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prix
    ax.plot(data.index, data['Close'], color='steelblue', linewidth=1.5, 
            label='S&P 500', alpha=0.7)
    
    # Anomalies Z-Score
    z_anomalies = data[data['ZScore_Anomaly'] == 1]
    ax.scatter(z_anomalies.index, z_anomalies['Close'], color='red', s=80, 
               marker='v', label=f'Z-Score (n={len(z_anomalies)})', zorder=5, alpha=0.7)
    
    # Anomalies Isolation Forest
    if_anomalies = data[data['IF_Anomaly'] == 1]
    ax.scatter(if_anomalies.index, if_anomalies['Close'], color='purple', s=60, 
               marker='s', label=f'Isolation Forest (n={len(if_anomalies)})', zorder=4, alpha=0.6)
    
    # Anomalies LOF
    lof_anomalies = data[data['LOF_Anomaly'] == 1]
    ax.scatter(lof_anomalies.index, lof_anomalies['Close'], color='green', s=40, 
               marker='D', label=f'LOF (n={len(lof_anomalies)})', zorder=3, alpha=0.5)
    
    ax.set_title('Comparaison des 3 méthodes de détection', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Prix de clôture ($)', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, '06_comparison_all_methods.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Sauvegardé : {path}")
    plt.close()


def generate_comparison_table(data):
    """
    Tableau de comparaison des méthodes
    """
    print("\n📊 Génération du tableau de comparaison...")
    
    results = {
        'Méthode': ['Z-Score', 'Isolation Forest', 'LOF'],
        'Anomalies détectées': [
            data['ZScore_Anomaly'].sum(),
            data['IF_Anomaly'].sum(),
            data['LOF_Anomaly'].sum()
        ],
        'Pourcentage': [
            f"{(data['ZScore_Anomaly'].sum() / len(data)) * 100:.2f}%",
            f"{(data['IF_Anomaly'].sum() / len(data)) * 100:.2f}%",
            f"{(data['LOF_Anomaly'].sum() / len(data)) * 100:.2f}%"
        ]
    }
    
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("📊 COMPARAISON DES MÉTHODES")
    print("="*60)
    print(df_results.to_string(index=False))
    
    # Concordance
    print("\n" + "="*60)
    print("🔍 CONCORDANCE ENTRE MÉTHODES")
    print("="*60)
    
    # Détectées par les 3
    all_three = data[(data['ZScore_Anomaly'] == 1) & 
                     (data['IF_Anomaly'] == 1) & 
                     (data['LOF_Anomaly'] == 1)]
    print(f"Détectées par les 3 méthodes : {len(all_three)}")
    
    # Détectées par 2
    two_methods = data[((data['ZScore_Anomaly'] == 1) & (data['IF_Anomaly'] == 1)) |
                       ((data['ZScore_Anomaly'] == 1) & (data['LOF_Anomaly'] == 1)) |
                       ((data['IF_Anomaly'] == 1) & (data['LOF_Anomaly'] == 1))]
    print(f"Détectées par au moins 2 méthodes : {len(two_methods)}")
    
    # Sauvegarder le tableau
    path = os.path.join(OUTPUT_DIR, 'comparison_table.csv')
    df_results.to_csv(path, index=False)
    print(f"\n💾 Tableau sauvegardé : {path}")


def generate_all_visualizations():
    """
    Génère toutes les visualisations
    """
    print("\n" + "="*60)
    print("📊 GÉNÉRATION DES VISUALISATIONS")
    print("="*60)
    
    # Charger les résultats
    data = load_results()
    
    # Créer les graphiques
    plot_price_evolution(data)
    plot_returns_distribution(data)
    plot_anomalies_zscore(data)
    plot_anomalies_if(data)
    plot_anomalies_lof(data)
    plot_comparison(data)
    
    # Tableau de comparaison
    generate_comparison_table(data)
    
    print("\n" + "="*60)
    print("✅ TOUTES LES VISUALISATIONS CRÉÉES")
    print("="*60)
    print(f"\n📁 Graphiques disponibles dans : {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all_visualizations()
    print("\n🎉 Module visualization.py fonctionne correctement !")