# IMDb Sentiment Analysis

A Streamlit dashboard for comparing eight deep-learning sentiment-analysis models trained on the IMDb 50k movie-review dataset.

The dashboard supports:

- Comparing reported accuracy and calculated metrics for four CNN and four LSTM models
- Running every available model on held-out IMDb reviews
- Entering a custom movie review and viewing live sentiment predictions
- Visualizing model performance, confidence scores, and architecture details

## Models

The app loads the following trained models:

| Family | Models | Input length |
| --- | --- | ---: |
| CNN | Flatten, Flatten + Dropout, GlobalMax + Dropout, GlobalMax + EarlyStopping | 100 tokens |
| LSTM | Basic, GloVe Embeddings, Own Embeddings v1, Own Embeddings v2 | 100 or 200 tokens |

The best reported accuracy is `87.4%` from `l1_lstm_own_weights.30.874.h5`.

## Requirements

- Python 3.10 or newer
- The files in this repository, including the trained model and dataset files
- A machine with enough memory to load TensorFlow and multiple Keras models

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On first run, the app may download the NLTK English stopword list and build `.tokenizer_cache.pkl` from the IMDb training data. The tokenizer is cached for later runs.

## Run the dashboard

With the virtual environment active:

```bash
streamlit run app.py
```

Or use the included launcher:

```bash
bash launch.sh
```

Then open `http://localhost:8501` in a browser.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard and inference pipeline |
| `requirements.txt` | Python dependencies |
| `launch.sh` | Convenience script for starting the app |
| `a1_IMDB_Dataset.csv` | IMDb training dataset used to build the tokenizer |
| `a3_IMDb_Unseen_Reviews.csv` | Held-out reviews used for evaluation in the dashboard |
| `a2_glove.6B.100d.txt` | GloVe 100-dimensional word embeddings |
| `*.h5` | Saved CNN and LSTM Keras models |
| `b_SentimentAnalysis_with_NeuralNetwork (1).ipynb` | Notebook containing the model-development workflow |

## Notes for GitHub

Several project assets are large. GitHub rejects regular Git files larger than 100 MB, and this project includes files above that limit, including the GloVe embeddings and some LSTM model files. Use [Git Large File Storage (Git LFS)](https://git-lfs.com/) for those assets before pushing:

```bash
git lfs install
git lfs track "*.h5"
git lfs track "a2_glove.6B.100d.txt"
git add .gitattributes
```

The app expects all model, dataset, and embedding files to remain in the same directory as `app.py`.

## License

No license has been specified yet. Add a license file before publishing if you want others to reuse or distribute this project.
