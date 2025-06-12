import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

# ======= Configuration générale =======
st.set_page_config(layout='wide', page_title="Bonheur dans le monde")

# ======= Présentation & Contexte =======
st.title("🌍 Tableau de bord : Analyse du Bonheur dans le Monde (2015–2019)")
st.markdown("**Source : Rapports mondiaux sur le bonheur (World Happiness Report)**")

st.markdown("""
Bienvenue sur ce tableau de bord interactif dédié à l'analyse du bonheur à travers le monde de 2015 à 2019,  
en nous appuyant sur les données issues des Rapports Mondiaux sur le Bonheur.

### Contexte  
Le bonheur d'un pays ne se limite pas à la richesse ou à l'économie. Il prend aussi en compte la santé, le soutien social, la liberté, la générosité et la confiance envers le gouvernement.  
Ce tableau de bord permet d'explorer ces différentes dimensions et de comprendre comment elles influencent le bien-être des populations.

### Problématique  
Comment expliquer les différences de bonheur entre pays ?  
Quels facteurs ont le plus d'impact ?  
Grâce à des graphiques et à un modèle mathématique, nous vous invitons à découvrir les réponses.

---
""")

# ======= Chargement et traitement des données =======
@st.cache_data
def load_data():
    years = [2015, 2016, 2017, 2018, 2019]
    dfs = []

    for year in years:
        df = pd.read_csv(f"{year}.csv")

        if year in [2015, 2016]:
            df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
            rename_map = {
                'country': 'pays',
                'region': 'region',
                'happiness_rank': 'rang',
                'happiness_score': 'score_bonheur',
                'economy_gdp_per_capita': 'pib_par_habitant',
                'family': 'soutien_social',
                'health_life_expectancy': 'esperance_vie',
                'freedom': 'liberte',
                'trust_government_corruption': 'confiance_gouv',
                'generosity': 'generosite',
                'dystopia_residual': 'residu_dystopie',
            }
        elif year == 2017:
            df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('.', '_')
            rename_map = {
                'country': 'pays',
                'happiness_rank': 'rang',
                'happiness_score': 'score_bonheur',
                'economy__gdp_per_capita_': 'pib_par_habitant',
                'family': 'soutien_social',
                'health__life_expectancy_': 'esperance_vie',
                'freedom': 'liberte',
                'trust__government_corruption_': 'confiance_gouv',
                'generosity': 'generosite',
                'dystopia_residual': 'residu_dystopie',
            }
        else:
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            rename_map = {
                'country_or_region': 'pays',
                'overall_rank': 'rang',
                'score': 'score_bonheur',
                'gdp_per_capita': 'pib_par_habitant',
                'social_support': 'soutien_social',
                'healthy_life_expectancy': 'esperance_vie',
                'freedom_to_make_life_choices': 'liberte',
                'generosity': 'generosite',
                'perceptions_of_corruption': 'confiance_gouv',
            }

        df.rename(columns=rename_map, inplace=True)
        df['year'] = year

        wanted_cols = ['pays', 'region', 'year', 'rang', 'score_bonheur', 'pib_par_habitant', 'soutien_social',
                       'esperance_vie', 'liberte', 'generosite', 'confiance_gouv']

        df = df[[col for col in wanted_cols if col in df.columns]]

        if 'region' not in df.columns:
            df['region'] = np.nan

        dfs.append(df)

    df_concat = pd.concat(dfs, ignore_index=True)
    df_concat['confiance_gouv'] = df_concat['confiance_gouv'].fillna(df_concat['confiance_gouv'].mean())
    return df_concat

df_all = load_data()

variables = ['pib_par_habitant', 'soutien_social', 'esperance_vie', 'liberte', 'generosite', 'confiance_gouv']

# ======= Sidebar : Filtres =======
st.sidebar.header("🔍 Filtres")
selected_year = st.sidebar.selectbox("Sélectionnez une année", sorted(df_all['year'].unique()))
regions_disponibles = sorted(df_all['region'].dropna().unique())
selected_region = st.sidebar.multiselect("Sélectionnez les régions", options=regions_disponibles, default=regions_disponibles)

df_filtre = df_all[(df_all['year'] == selected_year)]
if len(regions_disponibles) > 0:
    if df_filtre['region'].notna().any():
        df_filtre = df_filtre[df_filtre['region'].isin(selected_region)]

# ======= Visualisations =======
st.subheader(f"📊 Score de Bonheur - {selected_year}")

top10 = df_filtre.groupby('pays')['score_bonheur'].mean().sort_values(ascending=False).head(10)
pays_top1 = top10.index[0] if not top10.empty else "N/A"
score_top1 = top10.iloc[0] if not top10.empty else 0
moyenne_score = df_filtre['score_bonheur'].mean() if not df_filtre.empty else 0

texte_intro_top10 = f"En {selected_year}, le pays qui a obtenu le score de bonheur le plus élevé est **{pays_top1}** avec un score moyen de **{score_top1:.2f}**. "\
                    f"Pour donner un ordre d'idée, la moyenne de bonheur parmi les pays sélectionnés est de **{moyenne_score:.2f}**.\n\n"\
                    "Le graphique à gauche montre le top 10 des pays les plus heureux cette année-là, classés du plus heureux au moins heureux.\n\n"\
                    "Un score de bonheur élevé signifie que les habitants de ce pays se sentent généralement mieux dans leur vie, avec un bon équilibre entre leurs conditions économiques, sociales et personnelles."
st.markdown(texte_intro_top10)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🏆 Top 10 des pays les plus heureux")
    st.bar_chart(top10)

with col2:
    st.markdown("#### 📈 Analyse des relations entre variables")

    corr_vars = variables + ['score_bonheur']
    if not df_filtre.empty and len(df_filtre) > 1:
        corr_matrix = df_filtre[corr_vars].corr()
        corr_scores = corr_matrix['score_bonheur'].drop('score_bonheur')
        var_most_corr = corr_scores.abs().idxmax()
        val_most_corr = corr_scores[var_most_corr]
        texte_corr = f"Cette année, la variable la plus liée au score de bonheur est **{var_most_corr.replace('_', ' ')}** "\
                     f"avec un lien fort mesuré à **{val_most_corr:.2f}**.\n\n"\
                     "Cela signifie que lorsque cette variable augmente, le score de bonheur tend aussi à augmenter (ou diminuer), ce qui montre son influence importante sur le bien-être ressenti."
    else:
        corr_matrix = pd.DataFrame()
        texte_corr = "Pas assez de données pour calculer les liens entre variables cette année."

    st.markdown(texte_corr)

    if not corr_matrix.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr_matrix, annot=True, cmap='crest', ax=ax)
        st.pyplot(fig)

# ======= Scatterplots =======
st.subheader("🔬 Analyse détaillée des facteurs du bonheur")

texte_scatter_intro = f"Pour l'année {selected_year}, nous explorons comment chaque facteur (PIB, soutien social, espérance de vie, etc.) est lié au score de bonheur. "\
                      "Chaque graphique montre la relation entre le score de bonheur et une variable, ce qui aide à comprendre leur influence."
st.markdown(texte_scatter_intro)

for var in variables:
    if var in df_filtre.columns and len(df_filtre) > 1:
        corr = df_filtre[[var, 'score_bonheur']].corr().iloc[0, 1]
        sens = "positive" if corr > 0 else "négative"
        texte_var = f"Le lien entre le score de bonheur et **{var.replace('_', ' ')}** est **{sens}** avec une force de **{corr:.2f}**.\n\n"\
                    f"Sur le graphique, chaque point représente un pays. On observe que plus la variable augmente, plus le score de bonheur a tendance à {'augmenter' if corr>0 else 'diminuer'}, "\
                    "mais ce n’est pas une règle absolue, il peut y avoir des exceptions."
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_filtre, x=var, y='score_bonheur', ax=ax, color='royalblue')
        ax.set_title(f'Score de bonheur vs {var.replace("_", " ").capitalize()}')
        st.markdown(texte_var)
        st.pyplot(fig)

# ======= Régression Linéaire avec Pipeline =======
st.subheader("📐 Modèle mathématique pour comprendre le bonheur")

st.markdown("""
Pour aller plus loin, nous utilisons un modèle mathématique appelé **régression linéaire**.  
Ce modèle essaie de prédire le score de bonheur d'un pays en fonction des facteurs comme le PIB, le soutien social, etc.  
Il attribue à chaque facteur un poids (appelé coefficient) qui indique à quel point ce facteur influence le bonheur.

Une valeur positive signifie que ce facteur augmente le score de bonheur, une valeur négative signifie qu'il le diminue.
""")

X = df_all[variables]
y = df_all['score_bonheur']

X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modele = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('regression', LinearRegression())
])

modele.fit(X_train, y_train)
y_pred = modele.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

col3, col4 = st.columns(2)
with col3:
    st.metric("R² (qualité de la prédiction)", f"{r2:.3f}")
with col4:
    st.metric("RMSE (erreur moyenne)", f"{rmse:.3f}")

fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax.set_xlabel('Score réel')
ax.set_ylabel('Score prédit')
ax.set_title('Score réel vs Score prédit par le modèle')
st.pyplot(fig)

coefs = modele.named_steps['regression'].coef_
coef_df = pd.DataFrame({
    'Variable explicative': [v.replace("_", " ").capitalize() for v in variables],
    'Coefficient': coefs
}).sort_values(by='Coefficient', ascending=False)

st.markdown("#### 🧮 Importance des facteurs selon le modèle")
st.markdown("""
Le tableau ci-dessous montre l'importance relative des facteurs pour expliquer le bonheur, selon le modèle mathématique.  
Par exemple, un coefficient élevé pour le PIB signifie qu'une meilleure richesse par habitant est fortement associée à un meilleur bonheur.  
""")
st.dataframe(coef_df, use_container_width=True)

if not coef_df.empty:
    var_plus_impact = coef_df.iloc[0]['Variable explicative']
    coef_plus_impact = coef_df.iloc[0]['Coefficient']
    conclusion_modele = f"Selon le modèle, le facteur ayant le plus d’impact sur le bonheur est **{var_plus_impact}** "\
                       f"avec un effet positif estimé à **{coef_plus_impact:.3f}**. Cela confirme l'importance de ce facteur pour le bien-être des populations."
    st.markdown(conclusion_modele)

# ======= Conclusion =======
st.markdown("---")
st.header("🔚 Conclusion")

conclusion_text = (
    f"En {selected_year}, le pays le plus heureux est **{pays_top1}**, avec un score de {score_top1:.2f}. "
    "Cette analyse montre clairement que plusieurs facteurs jouent un rôle important dans le bien-être ressenti par les habitants :\n\n"
    "- Le PIB par habitant, qui mesure la richesse moyenne, contribue significativement.\n"
    "- Le soutien social, qui reflète l’entraide et la solidarité entre personnes.\n"
    "- L’espérance de vie, un indicateur de santé et de qualité de vie.\n\n"
    "Le modèle mathématique utilisé permet de quantifier l’importance relative de ces facteurs, confirmant ainsi leur rôle central.\n\n"
    "Nous vous encourageons à explorer ce tableau de bord pour mieux comprendre les différences entre pays, régions et années, et à réfléchir à ce qui rend un pays plus heureux."
)
st.markdown(conclusion_text)

# ======= Footer =======
st.markdown("---")
st.markdown("📘 *Projet Data Science – Analyse du Bonheur mondial (2015-2019)* | Réalisé avec Python & Streamlit")