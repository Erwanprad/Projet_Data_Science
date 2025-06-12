# Analyse du Bonheur dans le Monde (2015-2019)

## Auteur
- Nom : PRADEILLES Erwan

## Description du projet
Ce projet propose un tableau de bord interactif d'analyse du bonheur mondial, basé sur les données des Rapports Mondiaux sur le Bonheur de 2015 à 2019.  
Il permet d'explorer les facteurs influençant le score de bonheur des pays, comme le PIB par habitant, le soutien social, l'espérance de vie, la liberté, la générosité et la confiance dans le gouvernement.

## Dataset
Les données utilisées proviennent des fichiers CSV des Rapports Mondiaux sur le Bonheur pour les années 2015 à 2019.  
Chaque fichier contient des indicateurs socio-économiques et le score de bonheur pour différents pays.

## Instructions pour lancer le dashboard
1. Cloner ce dépôt GitHub :  
   `git clone https://github.com/Erwanprad/Projet_Data_Science.git`  
2. Se déplacer dans le dossier du projet :  
   `cd Projet_Data_Science`  
3. Installer les dépendances :  
   `pip install -r requirements.txt`  
4. Lancer le dashboard :  
   `streamlit run dashboard_app.py`  
5. Le dashboard s’ouvre dans votre navigateur par défaut.

## GIFs animés
*(Ajouter ici des GIFs démontrant le fonctionnement du dashboard, par exemple en les plaçant dans un dossier `docs` et en insérant des liens Markdown)*

## Conclusions principales
- Le pays avec le score de bonheur le plus élevé en 2019 est [Nom du pays], illustrant l’importance de facteurs comme le PIB par habitant, le soutien social et l’espérance de vie.  
- Les analyses montrent que ces facteurs socio-économiques sont fortement liés au bien-être général des populations.  
- Le modèle de régression linéaire confirme l’impact relatif des différentes variables sur le score de bonheur.  
- Ce tableau de bord permet une exploration intuitive des données par année et par région.

## Sources et Références
- Dataset : World Happiness Report (2015-2019)  
- Outils utilisés : Python, Streamlit, pandas, seaborn, scikit-learn  
- Méthodes : Analyse exploratoire, visualisations interactives, modèle de régression linéaire  
- Modèle de langage : ChatGPT (OpenAI) pour assistance dans la rédaction du code et documentation
