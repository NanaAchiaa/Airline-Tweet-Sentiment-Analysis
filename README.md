# Airline-Tweet-Sentiment-Analysis
This project analyzes sentiment in tweets about U.S. airlines using natural language processing (NLP) techniques and machine learning models.


## Objective
To classify customer tweets about major U.S. airlines into **positive**, **neutral**, or **negative** sentiment and identify which airlines receive the most complaints and praise.

## Dataset
- Source: [Kaggle - Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- ~14,000 tweets labeled as **positive**, **neutral**, or **negative**
- Additional metadata includes tweet location, airline name, and reasons for negative sentiment.

## Preprocessing
- Cleaned tweet text: removed punctuation, URLs, stopwords, applied lemmatization
- Handled missing values
- Converted text to numerical features using **TF-IDF**

## Models Used
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Support Vector Machine (SVM)**

Each model was tuned using **GridSearchCV** with cross-validation.

## Results
| Model               | Accuracy | F1 Score | Best Params                        |
|--------------------|----------|----------|------------------------------------|
| **KNN**             | 73.1%    | 72.5%    | `n_neighbors = 9`                  |
| **Logistic Regression** | 78.8%    | 77.7%    | `C = 1`                             |
| **SVM**             | **79.1%**    | **78.2%**    | `C = 10, kernel = 'rbf'`             |

## Visualizations
- Sentiment distribution
- Sentiment per airline
- Most common negative reasons
- Model performance comparison

## Future Improvements
- Use **Word2Vec** or **BERT embeddings** for deeper semantic understanding
- Deploy using **Streamlit** for interactive predictions
- Incorporate **SHAP** or **LIME** for model interpretability

## Demo
[Streamlit App Coming Soon ]

## Folder Structure
