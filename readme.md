# A5-Sentence-Embedding-with-BERT

## Overview
This project focuses on the development and analysis of sentence embedding techniques using BERT models. The primary objectives include:

1. **Training BERT from Scratch with Sentence Transformer**: Detailing the process involved in training a BERT model from scratch, specifically for sentence transformation tasks.
2. **Sentence Embedding with Sentence BERT**: Implementing and fine-tuning a Sentence BERT (S-BERT) model to generate sentence embeddings effectively.
3. **Evaluation and Analysis**: Conducting a comprehensive evaluation of our trained models against other pre-trained models, focusing on metrics such as cosine similarity, Spearman correlation, and standard NLP performance metrics (Precision, Recall, F1 Score).
4. **Text Similarity Web Application Development**: Developing a Dash-based  web application to demonstrate the practical application of the trained Sentence BERT model, allowing users to compute the similarity between two input sentences.

## Datasets Used

### BookCorpus
- **Source**: [Hugging Face Datasets](https://huggingface.co/datasets/bookcorpus)
- **Description**: A large collection of text from books across various genres, providing rich content for training language models and semantic analysis tasks.
- **Usage**: The first 10,000 samples from the 'train' split were utilized for model training.

### SNLI (Stanford Natural Language Inference)
- **Source**: [Hugging Face Datasets](https://huggingface.co/datasets/snli)
- **Description**: Contains sentence pairs annotated with entailment relations, foundational for natural language inference tasks.

### MNLI (MultiNLI)
- **Source**: [Hugging Face Datasets](https://huggingface.co/datasets/multi_nli)
- **Description**: Extends SNLI by offering sentence pairs across a broader range of genres, testing the model's domain generalization.

## Model Details

### Custom Model Configuration
- **Hyperparameters**:
  - `n_layers`: 6 (Increases depth for learning complex patterns)
  - `n_heads`: 8 (Allows the model to simultaneously focus on different parts of the sequence)
  - `d_model`: 768 (Determines the dimensionality of embeddings, affecting model capacity)
  - `d_ff`: 3072 (FeedForward dimension, typically set to 4x `d_model`)
  - `d_k` and `d_v`: 64 (Dimensions for key/query and value vectors in the attention mechanism)
  - `n_segments`: 2 (Segment types, used in tasks involving multiple sequences)

## Inference Cosine Similarity Analysis

### Similar Sentences
- **Custom Model**: Achieves very high cosine similarity scores (close to 1) for similar sentences, indicating excellent performance in capturing semantic similarity.
- **Pretrained Model**: Also shows very high cosine similarity (almost 1) for similar sentences, confirming its effectiveness in semantic similarity tasks.

### Dissimilar Sentences
- **Custom Model**: Maintains high cosine similarity scores with dissimilar sentences, which is counterintuitive and suggests the need for model refinement.
- **Pretrained Model**: Displays appropriately lower cosine similarity scores for dissimilar sentences, demonstrating better semantic distinction.

## Hyperparameter Impact Analysis

### Learning Rate
- **Impact**: Crucial for determining the optimization step size. Too high can cause overshooting, too low may lead to slow convergence.
- **Strategy**: Employ adaptive learning rate methods and consider schedules like warm-up or decay to fine-tune training.

### Batch Size
- **Impact**: Influences training stability and speed. Smaller sizes may improve generalization but introduce noise in gradient estimates.
- **Strategy**: Experiment with sizes and use gradient accumulation for larger effective batches on constrained hardware.

### Number of Epochs
- **Impact**: Determines the number of training cycles through the dataset. Balancing is key to avoid underfitting or overfitting.
- **Strategy**: Implement early stopping based on validation performance to prevent overfitting.

### Model Architecture Parameters (n_layers, n_heads, d_model, d_ff, d_k, d_v)
- **Impact**: These define the model's capacity. An optimal configuration is vital for learning complex patterns without overfitting.
- **Strategy**: Start with known configurations for similar tasks. Adjust based on validation performance and apply regularization techniques as necessary.

### Regularization Parameters
- **Impact**: Controls the model's generalization capabilities to prevent over-reliance on specific features.
- **Strategy**: Tune based on validation set performance. Adjust dropout rates in conjunction with model size and learning rate.


## Limitations and Proposed Improvements

- **Dataset Imbalance**: Exploring data augmentation techniques could help address imbalance and improve model robustness.
- **Computational Resources**: Utilizing more efficient architectures or cloud-based resources could alleviate computational constraints.
- **Hyperparameter Optimization**: Employing automated hyperparameter tuning methods might identify configurations that enhance model performance.
- **Model Generalization**: Further fine-tuning on diverse datasets or incorporating domain-specific pretraining could improve model generalization.


## Web Application Development

![App Image](app.png)


A Dash-based web application has been developed to showcase the Sentence BERT model's ability to calculate similarity scores between sentences, demonstrating the model's practical application in real-world scenarios.
