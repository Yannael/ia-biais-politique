# Generative AI political biases

Interface accessible sur HuggingFace à [https://huggingface.co/spaces/Yannael/gen-ia-biais-politique](https://huggingface.co/spaces/Yannael/gen-ia-biais-politique)

## Motivations

🧠 **Les opinions des IA diffèrent-elles selon la langue ?**

Interrogez Grok (x.AI, Elon Musk) sur l'affirmation _« Toute autorité devrait être mise en question »_ :

- En **français** : il **approuve**, au nom de la vigilance démocratique.
    
- En **anglais** : il **désapprouve**, invoquant les dangers d’un scepticisme généralisé, notamment envers les secours ou la science.
    

🔍 Cette interface vous permet d’explorer et comparer les **opinions de plusieurs chatbots** (Grok, ChatGPT, Mistral, DeepSeek) sur **62 questions de société** – et d’observer **comment leurs biais varient selon la langue**.

📊 Résultat ? Une tendance commune au **libertarianisme de gauche**, encore plus marquée en français – sauf chez Mistral, de façon surprenante.


## Génération des réponses

Vous pouvez re-générer l'ensemble des réponses avec la commande suivante:

```bash
run_all_models.sh
```

Les résultats seront sauvegardés dans les répertoire 'responses' et le fichier `results.json`.

Les modèles sont appelés via l’API OpenRouter. Vous devrez configurer votre clé API dans le fichier .env.

## Génération des boussoles politiques

Vous pouvez regénérer les graphiques des boussoles politiques avec la commande suivante :

```
python generate_plots.py
```

Les graphiques seront enregistrés dans le dossier `plots`.

## Inspiration

- [Liu, Y., Panwang, Y. & Gu, C. “Turning right”? An experimental study on the political value shift in large language models. Humanit Soc Sci Commun 12, 179 (2025).](https://www.nature.com/articles/s41599-025-04465-z)
- [David Rozado's work](https://davidrozado.substack.com/p/new-results-of-state-of-the-art-llms)
- [Political Compass](https://politicalcompass.org/)
- [TrackingAI](https://trackingai.io/)
- [SpeechMap](https://speechmap.ai/)

