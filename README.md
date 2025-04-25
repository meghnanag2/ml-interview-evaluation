# Speech-Based Machine Learning for Interview Outcome Prediction

This repository contains the complete implementation of a machine learning project focused on the prediction of interview outcomes using multimodal data. The project specifically targets the estimation of two annotated metrics — **Overall Performance** and **Excitement Level** — based on linguistic and prosodic cues from the [MIT Interview Dataset](https://people.csail.mit.edu/mnaim/data.html).


## Repository Contents


```
├── MLProjectFinal.ipynb               # Primary Jupyter notebook containing the full workflow
├── MITInterview_data
    ├── transcripts.csv                # Raw interview transcripts (interviewer + interviewee)
    ├── prosodic_features.csv         # Aggregated prosodic/acoustic feature set
    ├── scores.csv                    # Annotator-assigned interview outcome labels 
```

## Project Summary

This work presents an end-to-end pipeline for building **explainable machine learning models** that assess interview performance. The models integrate features extracted from both **verbal content** (transcripts) and **paralinguistic signals** (prosodic features such as pitch, energy, and speech rate). The key objectives include:

- Accurate prediction of human-annotated **Overall** and **Excited** scores
- Evaluation of classical models (Random Forests, MLPs) vs. transformer-based methods (GPT-2)
- Integration of **explainable AI** methods (e.g., SHAP) to ensure interpretability
- Experimental use of **prompt engineering** for zero-shot inference with GPT-2


## Methodology

### Data Preprocessing
- **Text cleaning** using spaCy: tokenization, lemmatization, stopword and numeric filtering
- **Speaker separation**: Isolation of interviewee responses
- **Prosodic feature aggregation** across multiple utterances

### Feature Engineering
- TF-IDF vectorization of transcripts
- Part-of-speech (POS) tag frequency extraction
- VADER sentiment scores
- Optional BERT embeddings (excluded from final model due to interpretability constraints)

### Modeling
- Feature selection using Pearson correlation and Mutual Information
- Model training via cross-validated **MLPRegressor** and **Random Forest Regressor**
- GPT-2 inference with prompt-based regression and explanation generation

### Evaluation
- **Pearson’s correlation (r)**
- **Mean Absolute Error (MAE)**
- **Relative Error (RE)**


## Key Findings

- **Multimodal models** consistently outperform single-modality approaches
- **Prosodic features** are particularly effective for predicting **Excitement**
- **Linguistic features** better capture **Overall performance**
- **GPT-2**, although not fine-tuned, produced reasonable predictions with structured prompts, highlighting its utility in **zero-shot learning** scenarios
- **SHAP** analysis revealed interpretable and contextually meaningful feature contributions


## Tools and Libraries

- [scikit-learn](https://scikit-learn.org/)
- [spaCy](https://spacy.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [SHAP](https://shap.readthedocs.io/)
- pandas, NumPy, matplotlib, seaborn, tqdm

