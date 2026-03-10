# Speech-Based Machine Learning for Interview Outcome Prediction

An end-to-end machine learning project that predicts job interview outcomes by analyzing both **what a candidate says** and **how they say it**.  

This system models interview performance using **speech transcripts** and **prosodic vocal features**, estimating two annotated metrics from the MIT Interview Dataset:

- **Overall Interview Performance**
- **Excitement / Engagement Level**

The project explores **multimodal machine learning**, combining Natural Language Processing and acoustic speech analysis, while also incorporating **explainable AI techniques** to understand why the models make certain predictions.

 The Overall Picture:

<img width="987" height="807" alt="image" src="https://github.com/user-attachments/assets/8204d37a-411a-4bd1-a0d6-162086a3caca" />

## Project Motivation

In real interviews, evaluation is influenced by multiple factors. Interviewers consider:

- the **content** of the answers
- the **clarity and structure** of communication
- the **confidence and enthusiasm** in the candidate’s voice

Traditional automated systems usually focus only on text or only on speech signals. This project investigates whether combining both **linguistic signals and prosodic signals** can improve prediction of interview outcomes.

The goal is to build models that not only predict interview scores, but also provide **interpretable insights into the behaviors that influence those scores**.



## Dataset

This project uses the **MIT Interview Dataset**, which contains recorded interviews along with expert annotations.

The dataset consists of three main components:

#### Transcripts
<img width="1412" height="503" alt="image" src="https://github.com/user-attachments/assets/43329db9-d358-4f5d-a185-6c3276325006" />

Text transcripts of the interview conversations between interviewer and candidate.

These transcripts capture the **linguistic content** of the interview.

Example:

Interviewer: Tell me about a project you worked on.  
Candidate: I recently worked on a machine learning system where I designed a model to predict user behavior...



#### Prosodic Features

<img width="1354" height="503" alt="image" src="https://github.com/user-attachments/assets/8593fc3a-b2c2-4956-8a9c-6710333e7f60" />

Quantitative acoustic features extracted from interview audio recordings.

These features capture **how the candidate speaks**, including vocal patterns that may reflect enthusiasm, confidence, or hesitation.

Examples of prosodic features include:

- pitch
- speaking rate
- intensity
- pause duration
- number of speech breaks



#### Annotated Interview Scores
<img width="220" height="503" alt="image" src="https://github.com/user-attachments/assets/654e2f45-5ac8-47ed-8648-a70999b4edde" />

Each interview is annotated by human evaluators who assign scores for:

**Overall Performance**  
A general measure of interview quality and response effectiveness.

**Excitement Level**  
An estimate of the candidate's energy, enthusiasm, and engagement.

These scores serve as the **target variables** for the machine learning models.



## Data Processing Pipeline

The project follows a structured data processing workflow.

First, transcript data is cleaned and transformed into linguistic features.  
Then, prosodic features are aggregated at the participant level.  
Finally, both feature groups are merged into a unified dataset used for model training.

The main stages of the pipeline are:

1. Transcript preprocessing
2. Prosodic feature aggregation
3. Feature engineering
4. Model training
5. Model evaluation
6. Explainability analysis



## Transcript Preprocessing

Transcript data is processed using **spaCy** to extract meaningful linguistic features.

The preprocessing steps include:

- tokenization
- lowercasing
- punctuation removal
- stopword filtering
- lemmatization

Additionally, the transcripts are separated so that only **candidate responses** are used for modeling.

From the cleaned transcripts, several linguistic features are extracted.

###### TF-IDF Features

Term Frequency–Inverse Document Frequency vectors are used to represent important words in the candidate responses.

TF-IDF highlights words that are important within a transcript but not overly common across all transcripts.

###### Part-of-Speech (POS) Features

Using spaCy, the frequency of grammatical structures such as nouns, verbs, and adjectives is extracted to capture patterns in communication style.

###### Sentiment Features

Sentiment scores are computed using **VADER sentiment analysis**, providing indicators of positive or negative emotional tone.



## Prosodic Feature Processing

Prosodic features are derived from the audio recordings of the interviews.

Since multiple speech segments may exist for a participant, these features are aggregated so that each participant has a **single feature vector** representing their vocal characteristics.

Examples of aggregated features include:

- mean pitch
- pitch variability
- speaking rate
- intensity
- pause frequency

These signals help capture aspects such as **fluency, confidence, and engagement** during speech.



## Feature Integration

After extracting both linguistic and prosodic features, the two feature sets are merged into a **multimodal dataset**.

Each row in the dataset corresponds to a participant and contains:

- transcript-based features
- prosodic speech features
- human-annotated interview scores

This combined representation enables the models to learn from both **language content and vocal delivery**.



## Modeling Approaches
<img width="1051" height="571" alt="Screenshot 2026-03-10 at 1 13 02 AM" src="https://github.com/user-attachments/assets/b1c04d9e-15c6-4a04-a1af-0aed818d157a" />


Multiple machine learning models are explored to evaluate different modeling strategies.

#### Random Forest Regressor

<img width="565" height="275" alt="Screenshot 2026-03-10 at 1 17 32 AM" src="https://github.com/user-attachments/assets/5e66f04b-b68b-4d14-b9bd-0d99f1a3357b" />

Random Forest is an ensemble learning method based on decision trees.

It works by training multiple trees and combining their predictions.

Advantages include:

- strong performance on tabular data
- robustness to noisy features

- ability to model nonlinear relationships

Random Forest models serve as a strong baseline for predicting interview outcomes.



#### Multi-Layer Perceptron (MLP)
<img width="504" height="491" alt="Screenshot 2026-03-10 at 1 18 16 AM" src="https://github.com/user-attachments/assets/ab40eb38-9f79-4adb-91ab-0164f5b7e843" />

A feedforward neural network is used to capture more complex interactions between features.

The MLP model consists of:

- an input layer representing engineered features
- hidden layers with nonlinear activation functions
- an output layer producing predicted interview scores

MLP models are particularly useful when combining **multiple feature modalities**.



#### GPT-2 Prompt-Based Scoring

In addition to traditional regression models, this project explores a **language model approach** using GPT-2.

Instead of training a supervised regression model, transcripts are provided to GPT-2 through structured prompts.

The prompt includes:

- example interview transcripts
- corresponding annotated scores
- a new transcript to evaluate

The model generates:

- a predicted score
- a natural language explanation describing the reasoning behind the prediction

This approach demonstrates how large language models can provide **human-readable feedback** alongside predictions.



## Model Evaluation

All regression models are evaluated using standard performance metrics.

#### Pearson Correlation

Measures the correlation between predicted scores and actual human-annotated scores.

Higher correlation indicates stronger alignment with human judgments.



#### Mean Absolute Error (MAE)

Measures the average magnitude of prediction errors.

Lower values indicate more accurate predictions.



#### Relative Error

A normalized metric that helps compare errors across different score ranges.



## Explainable Machine Learning

Understanding why a model makes a prediction is important for real-world deployment.

This project uses **SHAP (SHapley Additive exPlanations)** to interpret model predictions.

SHAP assigns a contribution value to each feature, indicating how much it influenced a specific prediction.

Through SHAP analysis, the model reveals interpretable behavioral signals.

For example:

Higher excitement scores often correlate with:

- greater pitch variability
- faster speaking rate
- positive sentiment

Lower excitement scores may correlate with:

- long pauses
- repetitive filler words
- lower vocal energy

Explainability helps ensure that predictions are **transparent and meaningful**.



## Key Insights


The experiments reveal several important observations. 

<img width="503" height="292" alt="image" src="https://github.com/user-attachments/assets/4aa2bf2a-9872-4916-a2cf-dce37458d2cf" />


Multimodal models that combine transcript and prosodic features consistently outperform models using only one modality.

Prosodic features play a particularly strong role in predicting **Excitement**, as vocal energy and speech rhythm are important indicators of engagement.

Linguistic features tend to capture **response structure and content quality**, which influence **Overall performance scores**.

Explainable AI methods provide additional insights into how different behavioral signals influence interview evaluations.



## Technologies Used

Python is used as the primary programming language for the project.

Key libraries include:

- scikit-learn for machine learning models
- spaCy for natural language processing
- VADER sentiment analysis for sentiment scoring
- Hugging Face Transformers for GPT-2 experiments
- SHAP for explainable machine learning
- pandas and NumPy for data manipulation
- matplotlib and seaborn for visualization

