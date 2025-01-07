# Signature Recognition and Word Prediction Projects

This repository contains the code, datasets, and results for two projects:

1. **Signature Recognition Using CNN and Manual Feature Extraction**
2. **Word Prediction Using LSTM**

---

## Projects Overview

### 1. **Signature Recognition Using CNN and Manual Feature Extraction**
This project explores signature recognition using two approaches:
- **Convolutional Neural Networks (CNN)** for automatic feature extraction and classification.
- **Artificial Neural Networks (ANN)** trained on manually extracted features using:
  - **Histogram of Oriented Gradients (HOG)**
  - **Scale-Invariant Feature Transform (SIFT)**

#### Key Steps:
1. **Preprocessing**: Signature images are cleaned and prepared.
2. **Segmentation**: Images are divided into individual signatures.
3. **Feature Extraction**:
   - CNN learns features directly from the image.
   - HOG and SIFT manually extract features for the ANN model.
4. **Classification**: Models classify the signatures.

#### Evaluation Metrics:
- Precision
- Recall
- F-measure
- Accuracy

#### Results:
- The CNN model demonstrates effective feature learning.
- However, the ANN model trained on HOG features outperforms CNN in terms of accuracy.
- This highlights the significance of feature selection and data preprocessing in signature recognition tasks.

---

### 2. **Word Prediction Using LSTM**
This project focuses on Natural Language Processing (NLP) using deep learning techniques. It implements a **word-level LSTM model** to predict the next word in a sequence, trained on Shakespeare’s plays.

#### Key Steps:
1. **Dataset**: A text corpus of Shakespeare's plays is used for training.
2. **Model**: LSTM is chosen for its ability to capture sequential dependencies.
3. **Integration**: A web-based interface allows users to input partial sentences and get real-time word predictions.

#### Applications:
- Text autocompletion
- Assistive technologies

#### Results:
The LSTM model effectively predicts contextually relevant words and is integrated into a user-friendly web interface.



