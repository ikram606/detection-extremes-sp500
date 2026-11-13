# Détection d'Anomalies & Modélisation des Événements Extrêmes (S&P 500)

Projet de Data Science et de statistique appliquée visant à identifier, quantifier et modéliser les événements extrêmes (*outliers* et chocs de queue de distribution) sur l'indice **S&P 500**.

---

## 1. Problématique & Approche Data Science

Les séries temporelles financières et macroéconomiques présentent des distributions non gaussiennes caractérisées par des queues épaisses (*fat tails*) et une forte asymétrie (*skewness*). Les modèles classiques basés sur l'hypothèse de normalité sous-estiment la fréquence et l'amplitude des événements extrêmes.

Ce projet met en œuvre un pipeline complet pour :
* **Analyser la distribution empirique** des rendements et des chocs sur le S&P 500 (tests de normalité, kurtosis, skewness, QQ-plots).
* **Détecter les anomalies et régimes extrêmes** via des seuils statistiques et des méthodes dédiées aux queues de distribution.
* **Quantifier le risque extrême** et visualiser la concentration temporelle des chocs (*volatility clustering*).

---

## 2. Méthodologie & Techniques Utilisées

* **Prétraitement & Feature Engineering :**
  * Extraction des séries historiques de l'indice S&P 500.
  * Calcul des rendements logarithmiques et arithmétiques, et des mesures de dispersion glissantes (*rolling volatility*).
* **Détection & Modélisation des Extrêmes :**
  * Identification des valeurs aberrantes par seuils statistiques (Z-score, écarts interquartiles IQR, quantiles extrêmes).
  * Analyse des queues de distribution (Théorie des Valeurs Extrêmes — EVT / loi de Pareto généralisée ou dépassements de seuil *Peaks Over Threshold*).
  * Mesures de risque de queue (*Value-at-Risk* — VaR, *Expected Shortfall* — ES).

---
## Structure du Projet
├── data/               # Données brutes et features extraites (sp500_features.csv)
├── results/            # Sorties et détections (all_detections.csv, graphiques)
├── config.py           # Paramètres globaux (seuils, fenêtres glissantes, période)
└── main.py             # Script principal d'exécution du pipeline

## 3. Stack Technique

* **Langage :** Python
* **Analyse de données & Calcul statistique :** Pandas, NumPy, SciPy
* **Visualisation :** Matplotlib, Seaborn

---

## 4. Installation et Exécution

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Exécuter le pipeline d'analyse et de détection
python main.py
