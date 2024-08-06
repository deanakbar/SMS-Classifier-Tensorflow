# SMS-Classifier-Tensorflow
A TensorFlow SMS classifier is a machine learning model designed to categorize SMS messages. Typically, this classifier is used to identify whether an SMS message is spam (unsolicited and unwanted messages) or not spam (legitimate messages).


## Impor the data :
   from google.colab import drive
   drive.mount("/content/gdrive", force_remount=True)
## Read the data : 
   import pandas as pd
   df = pd.read_csv("/content/gdrive/MyDrive/SMS Classifier/spam.csv", encoding='ISO-8859-1')
   df.head()


   
