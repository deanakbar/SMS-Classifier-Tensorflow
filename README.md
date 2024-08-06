# SMS-Classifier-Tensorflow

This project demonstrates how to build an SMS spam classifier using TensorFlow. The classifier identifies whether an SMS message is spam (unwanted) or not spam (legitimate), and demonstrates how to preprocess text data, build a neural network model, and train it to classify SMS messages as spam or ham using TensorFlow.

## Project Setup

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)
