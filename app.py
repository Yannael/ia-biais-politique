import gradio as gr
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import requests
from io import BytesIO

url = "https://github.com/Yannael/ia-biais-politique/blob/main/plots/political_compass_en.png?raw=true"
response = requests.get(url)
img_en = Image.open(BytesIO(response.content))

url = "https://github.com/Yannael/ia-biais-politique/blob/main/plots/political_compass_fr.png?raw=true"
response = requests.get(url)
img_fr = Image.open(BytesIO(response.content))

# Télécharger et enregistrer temporairement le GIF animé
url = "https://github.com/Yannael/ia-biais-politique/blob/main/plots/political_compass_animation.gif?raw=true"
gif_path = "temp.gif"
with open(gif_path, "wb") as f:
    f.write(requests.get(url).content)

# Charger les questions
questions = pd.read_csv('questions/questions_en_fr.csv')

def get_score(response):

    response_split = response.split("**")
    if len(response_split) > 1:
        score = response_split[1]
    else:
        response_split = response.split("*")
        if len(response_split) > 1:
            score = response_split[1]
        else:
            return None

    if pd.isna(score):
        return None
    agreement_map = {
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
    for phrase, num in agreement_map.items():
        if phrase == score:
            return num
    return None

def parse_responses(model, language):
    
    path = f"responses/responses_{language}_{model.replace('/', '_')}.csv"
    df = pd.read_csv(path)

    n = df.shape[1]

    for i in range(n):
        scores = df['response_'+str(i)].apply(get_score)
        df['score_'+str(i)] = scores

    df_scores = df[['score_'+str(i) for i in range(n)]]

    df['mean_scores'] = df_scores.mean(axis=1, skipna=True)
    df['std_scores'] = df_scores.std(axis=1, skipna=True)
    
    return df


def create_column_content(question_id, lang='fr'):
    # Obtenir le jeu de données de questions basé sur la langue
    question = questions['questions_fr' if lang == 'fr' else 'questions_en'].iloc[question_id]
    
    # Charger les réponses pour chaque modèle
    model_responses = {}
    model_scores = {}
    
    # Définir les familles de modèles et leurs modèles spécifiques
    models = ["x-ai_grok-beta", "openai_gpt-4o", "deepseek_deepseek-chat-v3-0324", "mistralai_mistral-large-2411"]
    
    # Charger les réponses depuis les fichiers CSV
    for model in models:
            try:
                df = parse_responses(model, lang)
                if not df.empty:
                    response = df[[col for col in df if col.startswith('response_')]]
                    score = df[[col for col in df if col.startswith('score_')]]
                    model_responses[model] = response.iloc[question_id].values
                    model_scores[model] = score.iloc[question_id].values
            except (FileNotFoundError, pd.errors.EmptyDataError):
                continue
    
    # Créer la visualisation
    fig = go.Figure()
    
    # Définir les couleurs pour les familles de modèles
    colors = {
        'openai': '#4285F4',
        'deepseek': '#34A853',
        'x-ai': '#EA4335',
        'mistralai': '#FBBC05'
    }
    
    # Traiter les données pour la visualisation
    for model_name, scores in model_scores.items():
        family = model_name.split('_')[0]
        model_display_name = model_name.split('_')[1]
        
        # Filtrer les valeurs None
        valid_scores = [s for s in scores if pd.notna(s)]
        
        if valid_scores:
            # Ajouter un graphique en boîte pour chaque modèle
            fig.add_trace(go.Box(
                x=valid_scores,
                name=model_display_name,
                marker_color=colors[family],
                boxpoints='all',  # Afficher tous les points
                jitter=0.3,  # Ajouter de la dispersion aux points
                pointpos=-1.8,  # Positionner les points à gauche de la boîte
                boxmean=True  # Afficher la ligne de moyenne
            ))
    
    # Mettre à jour la mise en page pour une meilleure visualisation
    fig.update_layout(
        title=dict(
            text="Distribution des scores (par modèle, de 1 à 10)",
            font=dict(size=16),
            xref='paper',
            x=0
        ),
        xaxis=dict(
            title='Score (1: Tout à fait d\'accord à 4: Pas du tout d\'accord)',
            range=[0, 5],  # Légèrement plus large que la plage de données pour une meilleure visibilité
            gridcolor='lightgray',
            zeroline=False
        ),
        yaxis=dict(
            title='Modèles',
            gridcolor='lightgray'
        ),
        showlegend=False,
        height=400,
        width=800,
        plot_bgcolor='white'
    )
    
    # Créer le contenu markdown avec les réponses
    md_content = f"### Réponses des modèles pour la question :\n**{question}**\n\n"
    
    for model_name, responses in model_responses.items():
        md_content += f"#### {model_name} \n"
        for i in range (len(responses)):
            md_content += f"#### Exécution {i}: \n\n {responses[i]}\n\n"
    
    return fig, md_content

def update_content(question_en, question_fr):
    # Obtenir les indices des questions
    idx_en = questions[questions['questions_en'] == question_en].index[0]
    idx_fr = questions[questions['questions_fr'] == question_fr].index[0]
    
    fig_en, answers_en_md = create_column_content(idx_en, 'en')
    fig_fr, answers_fr_md = create_column_content(idx_fr, 'fr')
    
    return fig_en, fig_fr, answers_en_md, answers_fr_md

css = """
h1, h3 {
    text-align: center;
    display:block;
}
"""

# Créer l'interface
with gr.Blocks(theme=gr.themes.Soft(), css=css) as interface:
    gr.Markdown('# Comment les biais politiques des chatbots diffèrent-ils')
    gr.Markdown('# entre l\'anglais et le français ?')
    
    gr.Markdown("---")
    gr.Markdown("## Aperçu")
    
    with gr.Row():          

        with gr.Column():
            gr.Markdown("""
🧠 **Les opinions des IA diffèrent-elles selon la langue ?**

Interrogez Grok (x.AI, Elon Musk) sur l'affirmation _« Toute autorité devrait être mise en question »_ :

- En **français** : il **approuve**, au nom de la vigilance démocratique.
    
- En **anglais** : il **désapprouve**, invoquant les dangers d’un scepticisme généralisé, notamment envers les secours ou la science.
    

🔍 Cette interface vous permet d’explorer et comparer les **opinions de plusieurs chatbots** (Grok, ChatGPT, Mistral, DeepSeek) sur **62 questions de société** – et d’observer **comment leurs biais varient selon la langue**.

📊 Résultat ? Une tendance commune au **libertarianisme de gauche**, encore plus marquée en français – sauf chez Mistral, de façon surprenante.

Auteur: [Yann-Aël Le Borgne](https://www.linkedin.com/in/yannaelb/)

Code source : [GitHub](https://github.com/Yannael/ia-biais-politique)

Inspiré par :
- [Liu, Y., Panwang, Y. & Gu, C. “Turning right”? An experimental study on the political value shift in large language models. Humanit Soc Sci Commun 12, 179 (2025).](https://www.nature.com/articles/s41599-025-04465-z)
- [Les travaux de David Rozado](https://davidrozado.substack.com/p/new-results-of-state-of-the-art-llms)
- [Boussole Politique](https://politicalcompass.org/)
- [TrackingAI](https://trackingai.io/)
- [SpeechMap](https://speechmap.ai/)
            """)

            
        with gr.Column():
            img_overview = gr.Image(value=gif_path, type="pil", label="Boussole Politique Animée")
            
    # Visualisations de la boussole politique
    gr.Markdown("---")
    gr.Markdown("## Boussole Politique")
    gr.Markdown("""
Ces deux cartes montrent la boussole politique des modèles.
            
            """)
    with gr.Row():          

        with gr.Column():
            compass_plot_en = gr.Image(value=img_en, type="pil", show_label=False)
        
        with gr.Column():
            compass_plot_fr = gr.Image(value=img_fr, type="pil", show_label=False)
    
    
    gr.Markdown("---")
    gr.Markdown("## Banque de questions et réponses des modèles")
    with gr.Row(): 
        # Obtenir les figures et le contenu initial
        initial_en_fig, initial_fr_fig, initial_en_md, initial_fr_md = update_content(
            questions['questions_en'].iloc[22],
            questions['questions_fr'].iloc[22]
        )

        with gr.Row():
            # Colonne de gauche pour les questions en anglais
            with gr.Column():
                gr.Markdown('### Questions en Anglais')
                dropdown_en = gr.Dropdown(
                choices=questions['questions_en'].tolist(),
                label='Sélectionner une question en anglais',
                value=questions['questions_en'].iloc[22]
                )
                plot_en = gr.Plot(value=initial_en_fig, label='Réponses en Anglais')
                answers_en = gr.Markdown(value=initial_en_md, label='Détails des Réponses')
    
            # Colonne de droite pour les questions en français
            with gr.Column():
                gr.Markdown('### Questions en Français')
                dropdown_fr = gr.Dropdown(
                    choices=questions['questions_fr'].tolist(),
                label='Sélectionner une question en français',
                value=questions['questions_fr'].iloc[22]
                )
                plot_fr = gr.Plot(value=initial_fr_fig, label='Réponses en Français')
                answers_fr = gr.Markdown(value=initial_fr_md, label='Détails des Réponses')
            
        # Mettre à jour le contenu quand les menus déroulants changent
        def sync_questions(selected_question, source_lang):
            # Trouver l'index de la question sélectionnée
            if source_lang == 'fr':
                idx = questions[questions['questions_fr'] == selected_question].index[0]
                corresponding_question = questions['questions_en'].iloc[idx]
            else:
                idx = questions[questions['questions_en'] == selected_question].index[0]
                corresponding_question = questions['questions_fr'].iloc[idx]
            
            # Obtenir le contenu pour les deux langues
            fig_en, fig_fr, md_en, md_fr = update_content(questions['questions_en'].iloc[idx], questions['questions_fr'].iloc[idx])
        
            return [
                corresponding_question,  # Mettre à jour l'autre menu déroulant
                fig_en, fig_fr,         # Mettre à jour les graphiques
                md_en, md_fr           # Mettre à jour le contenu markdown
            ]
    
        # Mettre à jour le contenu quand le menu déroulant français change
        dropdown_fr.change(
            fn=lambda q: sync_questions(q, 'fr'),
            inputs=[dropdown_fr],
            outputs=[dropdown_en, plot_en, plot_fr, answers_en, answers_fr]
        )
    
        # Mettre à jour le contenu quand le menu déroulant anglais change
        dropdown_en.change(
            fn=lambda q: sync_questions(q, 'en'),
            inputs=[dropdown_en],
            outputs=[dropdown_fr, plot_en, plot_fr, answers_en, answers_fr]
        )
    

if __name__ == '__main__':
    interface.launch(share=True)