# NLP

## 1. Neural Machine Translation 
Link :- https://github.com/vanshika230/LGMVIP_DataScienceIntern/blob/main/NMT.ipynb

Tech Stack :- Python, ScikitLearn, Tensorflow

--> About the project :- 

The code imports the necessary libraries and modules for data processing and modeling.Some initial data exploration and cleaning operations are performed, such as filtering data from a specific source, dropping columns, removing null values, converting text to lowercase, and applying regular expressions for text cleaning.

The code calculates the maximum sequence length for both English and Hindi sentences. Unique English and Hindi words are extracted and stored in separate lists.Tokenizers are created using Keras to convert the sentences into sequences of integers.

Model Architecture
  
    Embedding Layer: This layer plays a crucial role in mapping the input words to dense vectors of a fixed size. 
    By representing words in a continuous vector space, it captures semantic relationships, enabling better translation performance.

    LSTM (Long Short-Term Memory) Layer: LSTM is a type of recurrent neural network layer that excels at capturing long-range dependencies in the input sequence.     
    It processes the input word vectors and learns to encode contextual information, improving translation accuracy.

    RepeatVector Layer: To match the desired output sequence length, this layer repeats the output of the previous LSTM layer multiple times. 
    This repetition enables the model to generate a sequence of words based on the input, facilitating the translation process.
    
    LSTM Layer (with return_sequences=True): Similar to the previous LSTM layer, this layer is configured to return sequences instead of a single output. 
    It aids in decoding the input representation and generating the output sequence, contributing to the accurate translation of the text.

    Dense Layer: The final dense layer applies a softmax activation function to produce a probability distribution over the Hindi vocabulary. 
    It selects the most likely Hindi word for each position in the output sequence, ensuring the generated translations are linguistically appropriate.

Model Saving
Once the model is trained, we save it for future use. We accomplish this by using the save() method, which saves the entire model to a file. To save the model, we call model.save('NMTmodel'). 


## 2. News Articles Categorization:- 
Link:- https://github.com/vanshika230/iNeuron.ai/blob/main/NewsArticlesCategorization.ipynb

Techstack:- Python, scikitlearn, Tensorflow Keras, NLTK
Text Preprocessing: The code defines a function called process_text() for preprocessing the text data. It performs the following steps on each article's text:

Converts the text to lowercase and removes unnecessary characters such as "\r", "\n", and leading/trailing whitespaces.
Removes punctuation marks using regular expressions.
Tokenizes the text into words using the word_tokenize() function from nltk.
Removes stop words (common words like "and", "the", etc.) using the stopwords corpus from nltk.
Joins the filtered words back into a string.
The process_text() function is then applied to the "Text" column of the dataset using the apply() function and stores the processed text in a new column called "Text_parsed".

Label Encoding: To convert the categorical target variable ("Category") into numeric format, the code uses label encoding. It imports the LabelEncoder class from scikit-learn and applies it to the "Category" column. The encoded values are stored in a new column called "Category_target".

TF-IDF Vectorization: The code utilizes the TF-IDF (Term Frequency-Inverse Document Frequency) technique to convert the textual data into a numeric representation. It imports the TfidfVectorizer class from scikit-learn, which converts a collection of raw documents into a matrix of TF-IDF features. The code initializes the vectorizer with appropriate parameters and fits it on the training data using the fit_transform() function. It also transforms the testing data using the transform() function.

Model Training: Multiple Models with Grid Search: The code trains multiple models using scikit-learn's classifiers such as Random Forest Classifier, Support Vector Machine Classifier, and Multinomial Naive Bayes Classifier. Each model is trained using a Grid Search technique to find the best hyperparameters. The grid search is performed using the GridSearchCV class from scikit-learn. The code specifies a dictionary of parameter ranges to be searched and uses 3-fold cross-validation. The best model for each classifier is obtained by fitting the grid search on the training data.

Model Evaluation: The code evaluates the performance of each trained model on the testing data. For each model, it predicts the categories for the testing data using the predict() function and calculates various evaluation metrics such as accuracy, precision, recall, and F1 score using the classification_report() function from scikit-learn. The code also creates a confusion matrix for each model using the confusion_matrix() function and visualizes it using a heatmap (heatmap() function from seaborn).

## 3. Stock Price Prediction using LSTM deployed on Streamlit

Link :- https://github.com/vanshika230/Stock_Price_Prediction

Techstack:- Python, Tensorflow, Streamlit

Data Retrieval and Preprocessing:
We begin by importing the necessary libraries such as pandas, numpy, matplotlib, and quandl. We use the Quandl API to fetch the historical stock price data for AAPL. The data is stored in a pandas DataFrame, and irrelevant columns are dropped. The data is then split into training and testing sets.

Data Scaling:
To improve the performance of our LSTM model, we use MinMaxScaler to scale the training data to a range between 0 and 1.

Data Preparation:
We split the training data into input (x_train) and output (y_train) sequences. Each input sequence consists of 100 consecutive stock prices, and the corresponding output sequence contains the next stock price. This sliding window approach allows our model to learn patterns in the data.

Model Architecture:
Our LSTM model consists of four LSTM layers with varying numbers of units, followed by dropout layers to prevent overfitting, and a dense output layer. The architecture is as follows:

    LSTM layer with 50 units and return_sequences=True
    Dropout layer with a rate of 0.2
    LSTM layer with 60 units and return_sequences=True
    Dropout layer with a rate of 0.3
    LSTM layer with 80 units and return_sequences=True
    Dropout layer with a rate of 0.4
    LSTM layer with 120 units
    Dropout layer with a rate of 0.5
    Dense output layer with 1 unit
Model Compilation and Training:
We compile the model using the Adam optimizer and the mean squared error (MSE) loss function. The model is trained on the prepared training data for 50 epochs.

Results:
The training process shows a decreasing loss value, indicating that the model is learning the patterns in the data. After training, the model can be used to make predictions on the testing data.

Interactive Web Interface:
The web application is built using the Streamlit library, which provides a user-friendly interface for interacting with the model. Users can enter a stock ticker symbol in a text input field to retrieve the corresponding historical data and predictions. The descriptive statistics and visualizations are dynamically updated based on user input.

Visualizations:
The application presents two visualizations of the stock price data. Firstly, it displays a line chart comparing the closing price of the stock over time, along with the 100-day and 200-day moving averages. This chart helps users visualize the trends and patterns in the stock price data. Secondly, the application plots the original stock prices and the predicted stock prices on another line chart. This allows users to compare the actual prices with the model's predictions.

## 4. Alzheimer Detection from Live Speech 
Youtube link :- https://www.youtube.com/watch?v=Nbx6qjv7dMM&ab_channel=Vanshika

Tech stack:- StanfordNLP, spaCy, NLTK, Python, Tensorflow 

Neurocare is an app that aims to detect and provide support for Alzheimer's disease through early detection and wellness features.Using a voice memo recorded by the patient, Neurocare's trained model classifies speech as either AD (Alzheimer's disease) or non-AD. This classification is based on lexical analysis of grammar and pause durations. The model calculates a mental state examination score, aiding in reaching a diagnosis.

## 5. Oizys- Mental Health Chatbot

Youtube Link :- https://www.youtube.com/watch?v=W1e6aKjJFLo&ab_channel=Vanshika

Tech stack:- Python, Dialogflow, Node.js

This is a chatbot that will ask you about your problems indirectly and eventually find you solutions. The bot has been designed to generate empathy and positive emotions. I trained it using dialogflow agents to identify sentiments of the speaker by keywords and provide approproate solutions. 



