# Generative AI political biases

Check out the interface on HuggingFace: https://huggingface.co/spaces/Yannael/gen-ia-biais-politique

## How it works

The interface allows to compare the political compass of four models (OpenAI GPT-4o, DeepSeek DeepSeek-chat-v3-0324, X-ai Grok-beta, MistralAI Mistral-large-2411) on a set of 62 political questions from the [Political Compass](https://politicalcompass.org/).

The political compass is computed by averaging the scores of the models on each question.

The interface also allows to select a question and see the responses of the models for this question.

## Re-run the analysis

You can re-run the analysis by running the following command:

```bash
run_all_models.sh
```

The results will be saved in the `results.json` file.

Models are called through the [OpenRouter API](https://openrouter.ai/). You will need to set up your API key in the `.env` file.

## Generate political compasses

You can generate the political compasses plots by running the following command:

```bash
python generate_plots.py
```

The plots will be saved in the `plots` folder.

## Inspiration

- [Liu, Y., Panwang, Y. & Gu, C. “Turning right”? An experimental study on the political value shift in large language models. Humanit Soc Sci Commun 12, 179 (2025).](https://www.nature.com/articles/s41599-025-04465-z)
- [David Rozado's work](https://davidrozado.substack.com/p/new-results-of-state-of-the-art-llms)
- [Political Compass](https://politicalcompass.org/)
- [TrackingAI](https://trackingai.io/)
- [SpeechMap](https://speechmap.ai/)

