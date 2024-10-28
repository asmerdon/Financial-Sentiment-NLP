# Sentiment Analysis of Financial News Tweets

This project performs sentiment analysis on financial news tweets by implementing and comparing traditional NLP techniques with transformer-based models.

## Overview

### Traditional NLP Approach

- **Data Preprocessing**: Cleaned and normalised text data.
- **Feature Extraction**: Used TF-IDF vectorisation.
- **Model Training**: Trained an SVM classifier with class weighting.
- **Cross-Validation**: Implemented stratified k-fold cross-validation.
- **Evaluation**: Measured accuracy, precision, recall, and F1-score.

### Transformer-Based Approach

- **Model Selection**: Pre-trained DistilBERT model.
- **Tokenization**: Used the model's tokenizer.
- **Fine-Tuning**: Fine-tuned on the dataset.
- **Training**: Leveraged Hugging Face Trainer API.
- **Evaluation**: Generated classification reports.

## Technologies & Tools

- **Programming Language**: Python
- **Libraries**:
  - **Traditional NLP**: `scikit-learn`, `NLTK`
  - **Transformer Models**: `transformers`, `datasets`
  - **Deep Learning Framework**: `PyTorch`

## How to Run

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/financial-sentiment-analysis.git
    ```

2. **Install Dependencies**:

    ```bash
    cd financial-sentiment-analysis
    pip install -r requirements.txt
    ```

3. **Download the Dataset**:

    ```python
    from datasets import load_dataset
    dataset = load_dataset('zeroshot/twitter-financial-news-sentiment')
    ```

4. **Run the Notebooks**:

   - **Traditional NLP Model**: `traditional_nlp_sentiment_analysis.ipynb`
   - **Transformer-Based Model**: `transformer_sentiment_analysis.ipynb`

## Results & Analysis

- **Performance Comparison**:
  - Transformer-based model outperformed the traditional SVM classifier.
  - Better performance on minority classes.

- **Insights**:
  - Transformer models capture contextual nuances better.
  - Traditional models are faster but may struggle with complex language patterns.

## License

This project is licensed under the MIT License.
