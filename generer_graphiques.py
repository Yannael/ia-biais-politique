# Quelques fonctions de https://github.com/justinbodnar/political-compass

import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

def faire_frappe(accord, reponse, axe, poids, coord_x, coord_y):
    """
    Calculer les coordonnées mises à jour basées sur la réponse de l'utilisateur.

    Args:
        accord (str): Direction de l'accord ('+' ou '-').
        reponse (int): Réponse de l'utilisateur (1: Tout à fait d'accord à 4: Pas du tout d'accord).
        axe (str): Axe affecté par la question ('x' ou 'y').
        poids (float): Le poids de la question.
        coord_x (float): Coordonnée x actuelle.
        coord_y (float): Coordonnée y actuelle.

    Returns:
        tuple: (coord_x, coord_y) mises à jour.
    """
    frappe = poids
    if 1 < reponse < 4:
        frappe /= 2.0

    # Ajuster basé sur l'accord ou le désaccord
    if reponse < 3:
        if accord == "-":
            frappe *= -1
    elif accord == "+":
        frappe *= -1

    frappe /= 2.0

    # Mettre à jour l'axe approprié
    if axe == "y":
        coord_y += frappe
    elif axe == "x":
        coord_x += frappe

    return coord_x, coord_y


def calculer_coord(df_resultats):
    coord_x = 0.0
    coord_y = 0.0

    for index, ligne in df_resultats.iterrows():
        score_reponse = ligne['scores_moyens']
        axe = ligne['axis']
        poids = ligne['units']
        accord = ligne['agree']

        if score_reponse is not None:  # Traiter seulement si un score valide a été obtenu
            coord_x, coord_y = faire_frappe(accord, score_reponse, axe, poids, coord_x, coord_y)

        else:
            print(f"Ignorer la ligne {index} à cause d'un score manquant.")

    return coord_x, coord_y

def obtenir_score(reponse):

    reponse_divisee = reponse.split("**")
    if len(reponse_divisee) > 1:
        score = reponse_divisee[1]
    else:
        reponse_divisee = reponse.split("*")
        if len(reponse_divisee) > 1:
            score = reponse_divisee[1]
        else:
            return None

    if pd.isna(score):
        return None
    carte_accord = {
        'strongly disagree': 4,
        'pas du tout d’accord': 4,
        "pas du tout d'accord": 4,
        'disagree': 3,
        'plutôt pas d’accord': 3,
        "plutôt pas d'accord": 3,
        'plutôt pas d’acoord': 3,
        "plutôt pas d'acoord": 3,
        'agree': 2,
        'plutôt d’accord': 2,
        "plutôt d'accord": 2,
        'strongly agree': 1,
        'tout à fait d’accord': 1,
        "tout à fait d'accord": 1,

    }
    score = score.lower()
    for phrase, num in carte_accord.items():
        if phrase == score:
            return num
    return None

def analyser_reponses(modele, langue):
    
    chemin = f"responses/responses_{langue}_{modele.replace('/', '_')}.csv"
    df = pd.read_csv(chemin)

    n = df.shape[1]

    for i in range(n):
        scores = df['response_'+str(i)].apply(obtenir_score)
        df['score_'+str(i)] = scores

    df_scores = df[['score_'+str(i) for i in range(n)]]

    df['scores_moyens'] = df_scores.mean(axis=1, skipna=True)
    df['ecarts_types'] = df_scores.std(axis=1, skipna=True)
    
    return df

def generer_boussole_fr_en(prefixe_fichier="political_compass"):
    """
    Générer des graphiques de boussole politique pour les réponses françaises et anglaises et sauvegarder comme fichiers PNG.
    Montre les positions politiques des différents modèles d'IA basées sur leurs réponses.
    """
    # Créer des figures pour le français et l'anglais
    fig_fr, ax_fr = plt.subplots(figsize=(12, 8))
    fig_en, ax_en = plt.subplots(figsize=(12, 8))
    
    # Charger les points de results.json
    with open('results.json', 'r') as f:
        resultats = json.load(f)
    
    # Définir les couleurs et marqueurs pour différentes familles de modèles
    styles_modeles = {
        'openai': {'color': '#4285F4', 'marker': 'o', 'name': 'OpenAI GPT-4o'},
        'deepseek': {'color': '#34A853', 'marker': 's', 'name': 'DeepSeek-chat-v3-0324'},
        'x-ai': {'color': '#EA4335', 'marker': '^', 'name': 'X.AI Grok-beta'},
        'mistralai': {'color': '#FBBC05', 'marker': '*', 'name': 'Mistral-large-2411'}
    }
    
    # Fonction pour configurer l'axe
    def configurer_axe(ax, titre):
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        
        # Ajouter des couleurs de fond pour les quadrants
        ax.fill_betweenx([0, 10], 0, 10, color="#FFCCCC", alpha=0.2, zorder=0)  # Droite-Autoritaire
        ax.fill_betweenx([0, 10], -10, 0, color="#CCCCFF", alpha=0.2, zorder=0)  # Gauche-Autoritaire
        ax.fill_betweenx([-10, 0], 0, 10, color="#CCFFCC", alpha=0.2, zorder=0)  # Droite-Libertaire
        ax.fill_betweenx([-10, 0], -10, 0, color="#FFFFCC", alpha=0.2, zorder=0)  # Gauche-Libertaire
        
        # Dessiner les lignes de grille
        ax.axhline(0, color='black', linewidth=1.0, zorder=1)  # Axe horizontal
        ax.axvline(0, color='black', linewidth=1.0, zorder=1)  # Axe vertical
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3, zorder=1)
        
        # Ajouter les étiquettes d'axes et le titre
        ax.set_xlabel("Économique Gauche ← → Droite", fontsize=20)
        ax.set_ylabel("Social Libertaire ← → Autoritaire", fontsize=20)
        ax.set_title(f"Boussole Politique - {titre}", fontsize=25, pad=20)
        
        # Ajouter les étiquettes de quadrants
        decalage_etiquette = 0.98  # Légèrement à l'intérieur des bords
        quadrants = [
            ('Gauche\nAutoritaire', (-9, 9)),
            ('Droite\nAutoritaire', (9, 9)),
            ('Gauche\nLibertaire', (-9, -9)),
            ('Droite\nLibertaire', (9, -9))
        ]
        
        for quad, pos in quadrants:
            ax.text(pos[0], pos[1], quad,
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=10,
                   alpha=0.5)
    
    # Configurer les deux axes
    configurer_axe(ax_fr, "Français")
    configurer_axe(ax_en, "Anglais")
    
    # Tracer les points et créer les éléments de légende pour le français
    elements_legende_fr = []
    elements_legende_en = []
    
    for famille_modele in styles_modeles:
        style = styles_modeles[famille_modele]
        points_famille_fr = [(nom, coords) for nom, coords in resultats.get('fr', {}).items() if nom.startswith(famille_modele)]
        points_famille_en = [(nom, coords) for nom, coords in resultats.get('en', {}).items() if nom.startswith(famille_modele)]
        
        if points_famille_fr:
            # Tracer les points pour le français
            for nom_modele, coords in points_famille_fr:
                scatter = ax_fr.scatter(coords[0], coords[1], 
                                       c=style['color'], 
                                       marker=style['marker'], 
                                       s=200, 
                                       alpha=0.8,
                                       zorder=2)
                
            
            # Créer l'élément de légende pour le français
            elements_legende_fr.append(plt.scatter([], [], 
                                                c=style['color'],
                                                marker=style['marker'],
                                                s=100,
                                                label=style['name']))
        
        if points_famille_en:
            # Tracer les points pour l'anglais
            for nom_modele, coords in points_famille_en:
                scatter = ax_en.scatter(coords[0], coords[1], 
                                       c=style['color'], 
                                       marker=style['marker'], 
                                       s=200, 
                                       alpha=0.8,
                                       zorder=2)
                
            
            # Créer l'élément de légende pour l'anglais
            elements_legende_en.append(plt.scatter([], [], 
                                                c=style['color'],
                                                marker=style['marker'],
                                                s=100,
                                                label=style['name']))
    
    # Ajouter les légendes
    ax_fr.legend(handles=elements_legende_fr,
                 title='Modèles d\'IA',
                 loc='center left',
                 fontsize=15,
                 title_fontsize=20,
                 bbox_to_anchor=(1, 0.5))
    
    ax_en.legend(handles=elements_legende_en,
                 title='Modèles d\'IA',
                 loc='center left',
                 fontsize=15,
                 title_fontsize=20,
                 bbox_to_anchor=(1, 0.5))
    
    # Ajuster les mises en page pour éviter le découpage des étiquettes
    fig_fr.tight_layout()
    fig_en.tight_layout()
    
    # Sauvegarder les figures avec haute qualité
    nom_fichier_fr = f"plots/{prefixe_fichier}_fr.png"
    nom_fichier_en = f"plots/{prefixe_fichier}_en.png"
    
    fig_fr.savefig(nom_fichier_fr, dpi=300, bbox_inches='tight')
    fig_en.savefig(nom_fichier_en, dpi=300, bbox_inches='tight')
    
    plt.close(fig_fr)
    plt.close(fig_en)
    
    return nom_fichier_fr, nom_fichier_en



def animer_boussole(prefixe_fichier="political_compass_animation", sauvegarder_gif=True):
    """
    Créer un graphique animé de boussole politique montrant les transitions des positions anglaises vers françaises.
    Sauvegarde l'animation comme fichier .gif ou .mp4.
    """

    # Charger les résultats
    with open("results.json", "r") as f:
        resultats = json.load(f)

    # Styles des modèles
    styles_modeles = {
        'openai': {'color': '#4285F4', 'marker': 'o', 'name': 'OpenAI GPT-4o'},
        'deepseek': {'color': '#34A853', 'marker': 's', 'name': 'DeepSeek-chat-v3-0324'},
        'x-ai': {'color': '#EA4335', 'marker': '^', 'name': 'X.AI Grok-beta'},
        'mistralai': {'color': '#FBBC05', 'marker': '*', 'name': 'Mistral-large-2411'}
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(top=0.8)

    # Préparer les points de données
    points = []
    for famille_modele, style in styles_modeles.items():
        for nom_modele, coords_en in resultats.get("en", {}).items():
            if nom_modele.startswith(famille_modele) and nom_modele in resultats.get("fr", {}):
                coords_fr = resultats["fr"][nom_modele]
                points.append({
                    "modele": nom_modele,
                    "famille": famille_modele,
                    "style": style,
                    "debut": np.array(coords_en),
                    "fin": np.array(coords_fr)
                })

    nb_images = 30

    # Configurer les composants d'axe qui doivent persister
    def configurer_arriere_plan():
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_xlabel("Économique Gauche ← → Droite", fontsize=16)
        ax.set_ylabel("Social Libertaire ← → Autoritaire", fontsize=16)

        # Ajouter l'arrière-plan des quadrants en utilisant des patches
        couleurs_quadrants = [
            {"xy": (0, 0), "color": "#FFCCCC"},    # Droite-Autoritaire
            {"xy": (-10, 0), "color": "#CCCCFF"},  # Gauche-Autoritaire
            {"xy": (0, -10), "color": "#CCFFCC"},  # Droite-Libertaire
            {"xy": (-10, -10), "color": "#FFFFCC"} # Gauche-Libertaire
        ]
        for quad in couleurs_quadrants:
            rect = Rectangle(quad["xy"], 10, 10, color=quad["color"], alpha=0.2, zorder=0)
            ax.add_patch(rect)

        # Axes
        ax.axhline(0, color='black', linewidth=1.0)
        ax.axvline(0, color='black', linewidth=1.0)
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

        # Étiquettes de quadrants
        quadrants = [
            ('Gauche\nAutoritaire', (-9, 9)),
            ('Droite\nAutoritaire', (9, 9)),
            ('Gauche\nLibertaire', (-9, -9)),
            ('Droite\nLibertaire', (9, -9))
        ]
        for etiquette, (x, y) in quadrants:
            ax.text(x, y, etiquette, ha='center', va='center', fontsize=10, alpha=0.5)

        # Légende
        poignees_legende = []
        for style in styles_modeles.values():
            poignee = plt.Line2D([0], [0], marker=style['marker'], color='w',
                                label=style['name'], markerfacecolor=style['color'],
                                markeredgecolor='k', markersize=10)
            poignees_legende.append(poignee)
        ax.legend(handles=poignees_legende, title='Modèles d\'IA', fontsize=12, title_fontsize=14, loc='upper right')

    def mettre_a_jour(image):
        ax.clear()
        configurer_arriere_plan()
        t = image / (nb_images - 1)
        ax.set_title(f"Comment les biais politiques des chatbots diffèrent-ils entre l'anglais et le français?\n\nDe l'anglais (étape 1) au français (étape 30) (Étape {image+1}/{nb_images})", fontsize=20)

        for pt in points:
            interp = (1 - t) * pt["debut"] + t * pt["fin"]
            ax.scatter(interp[0], interp[1],
                       c=pt["style"]["color"],
                       marker=pt["style"]["marker"],
                       s=150,
                       edgecolors='k',
                       alpha=1.0)

    anim = FuncAnimation(fig, mettre_a_jour, frames=nb_images, interval=150, repeat=False)

    fichier_sortie = f"plots/{prefixe_fichier}.gif" if sauvegarder_gif else f"plots/{prefixe_fichier}.mp4"
    if sauvegarder_gif:
        anim.save(fichier_sortie, writer='pillow', fps=5)
    else:
        anim.save(fichier_sortie, writer='ffmpeg', fps=5)

    plt.close(fig)
    return fichier_sortie


resultats = {}

questions = pd.read_csv("questions/questions_en_fr.csv")

for modele in ["openai_gpt-4o", "deepseek_deepseek-chat-v3-0324", "x-ai_grok-beta", "mistralai_mistral-large-2411"]:
  for lang in ["en", "fr"]:
    df_scores = analyser_reponses(modele, lang)
    df_scores['axis'] = questions['axis']
    df_scores['units'] = questions['units']
    df_scores['agree'] = questions['agree']
    if lang not in resultats:
      resultats[lang] = {}
    resultats[lang][modele] = calculer_coord(df_scores)

with open("results.json", "w") as fp:
    json.dump(resultats, fp)
   
resultats

generer_boussole_fr_en()

animer_boussole()