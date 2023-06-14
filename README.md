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


# 2. News Articles Categorization:- 
## Determining the best dataset to use :- https://github.com/vanshika230/iNeuron.ai/blob/main/text_clustering_and_visualization.ipynb

Techstack :- Python, plotly, matplotlib, seaborn, numpy

Data Loading: The code starts by importing the necessary libraries and loading the BBC News dataset into a pandas DataFrame. The dataset contains two columns: "category" and "text", where "category" represents the actual category of the news article, and "text" contains the textual content of the articles.

Feature Extraction: TF-IDF Vectorization: To perform clustering, the code uses the TfidfVectorizer from scikit-learn to convert the text into a numerical representation. It initializes the vectorizer and applies it to the "text" column of the DataFrame, creating a matrix of TF-IDF features. The shape of the feature matrix is printed to verify the extraction process.

Applying K-Means Clustering: The code uses the KMeans algorithm from scikit-learn to perform K-Means clustering on the extracted features. It initializes the KMeans object with the desired number of clusters (in this case, 5), the number of times the algorithm will be run with different centroid seeds, and the random state for reproducibility. The algorithm is fitted on the features to assign each news article to a cluster, and the cluster labels are added to the DataFrame.

Mapping Clusters to Categories: The code analyzes the clustering results for each category by counting the number of samples assigned to each cluster. For each category, it identifies the cluster with the highest number of samples and maps that cluster number to the corresponding category.

Assigning Clustered Category Labels: The code adds a new column to the DataFrame called "clustered_category" by mapping the cluster labels to their corresponding category labels using the mapping obtained in the previous step.

Accuracy Calculation: The code compares the actual category labels with the clustered category labels and calculates the overall accuracy of the clustering by computing the percentage of correctly assigned categories.

Visualization using PCA: To visualize the clustering results in a 2D space, the code uses Principal Component Analysis (PCA) to reduce the dimensionality of the features to two components. It initializes the PCA object, applies it to the feature matrix, and plots the data points in a scatter plot. The left graph represents the actual categories, while the right graph shows the clusters obtained from K-Means clustering.
## News articles Categorization :- https://github.com/vanshika230/iNeuron.ai/blob/main/NewsArticlesCategorization.ipynb

Techstack:- Python, scikitlearn

Importing Required Libraries: The initial step of the code is to import the necessary libraries such as pandas, matplotlib, pickle, seaborn, nltk, wordcloud, and scikit-learn modules for data manipulation, visualization, natural language processing, and machine learning.

Loading and Exploring Data: The code reads the dataset from a CSV file named "BBC News.csv" using the pandas library. The dataset contains articles with columns such as ArticleId, Text, and Category. The code then displays the first few rows of the dataset using the head() function to provide a glimpse of the data.

Understanding Features and Target Variables: The code explores the target variable, which is the "Category" column, to understand the unique categories in the dataset. It prints the unique categories using the unique() function. The code also checks the shape and data types of the dataset using the shape and dtypes attributes of the pandas DataFrame.

Checking for NULL Values: To ensure the quality of the dataset, the code checks for any missing values or NULL values in the dataset using the isnull().any() function. It prints a Boolean value indicating whether there are any missing values in the dataset.

Countplot of Target Variable: The code generates a countplot using the seaborn library to visualize the distribution of the target variable (Category). It displays the count of articles in each category using the countplot() function.

Feature Engineering: News Length: The code calculates the length of each news article by counting the number of characters in the "Text" column. It adds a new column called "News_length" to the dataset, which contains the length of each news article. It then visualizes the distribution of news lengths using a distribution plot (distplot() function from seaborn).

Text Preprocessing: The code defines a function called process_text() for preprocessing the text data. It performs the following steps on each article's text:

Converts the text to lowercase and removes unnecessary characters such as "\r", "\n", and leading/trailing whitespaces.
Removes punctuation marks using regular expressions.
Tokenizes the text into words using the word_tokenize() function from nltk.
Removes stop words (common words like "and", "the", etc.) using the stopwords corpus from nltk.
Joins the filtered words back into a string.
The process_text() function is then applied to the "Text" column of the dataset using the apply() function and stores the processed text in a new column called "Text_parsed".

Label Encoding: To convert the categorical target variable ("Category") into numeric format, the code uses label encoding. It imports the LabelEncoder class from scikit-learn and applies it to the "Category" column. The encoded values are stored in a new column called "Category_target".

Saving Processed Data: The code saves the processed dataset with the added columns as a new CSV file named "BBC_News_processed.csv" using the to_csv() function.

Splitting the Data: The code splits the dataset into training and testing sets using the train_test_split() function from scikit-learn. It splits the "Text_parsed" column as the feature variable (X) and the "Category_target" column as the target variable (y). The training set contains 80% of the data, while the testing set contains the remaining 20%.

Feature Extraction: TF-IDF Vectorization: The code utilizes the TF-IDF (Term Frequency-Inverse Document Frequency) technique to convert the textual data into a numeric representation. It imports the TfidfVectorizer class from scikit-learn, which converts a collection of raw documents into a matrix of TF-IDF features. The code initializes the vectorizer with appropriate parameters and fits it on the training data using the fit_transform() function. It also transforms the testing data using the transform() function.

Model Training: Multiple Models with Grid Search: The code trains multiple models using scikit-learn's classifiers such as Random Forest Classifier, Support Vector Machine Classifier, and Multinomial Naive Bayes Classifier. Each model is trained using a Grid Search technique to find the best hyperparameters. The grid search is performed using the GridSearchCV class from scikit-learn. The code specifies a dictionary of parameter ranges to be searched and uses 3-fold cross-validation. The best model for each classifier is obtained by fitting the grid search on the training data.

Model Evaluation: The code evaluates the performance of each trained model on the testing data. For each model, it predicts the categories for the testing data using the predict() function and calculates various evaluation metrics such as accuracy, precision, recall, and F1 score using the classification_report() function from scikit-learn. The code also creates a confusion matrix for each model using the confusion_matrix() function and visualizes it using a heatmap (heatmap() function from seaborn).

Saving the Trained Models: The code saves each trained model as serialized objects using the pickle library. It creates separate files for each model (e.g., "random_forest_model.pkl", "svm_model.pkl", "naive_bayes_model.pkl") and dumps the model objects into the respective files using the dump() function.

Summary and Conclusion: The code concludes by printing a summary of the performance metrics for each model and provides a brief summary of the results obtained. It also prints a message indicating the successful execution of the code. 

# Stock Price Prediction using LSTM

Link :- https://github.com/vanshika230/Stock_Price_Prediction

Techstack:- Python, Tensorflow, Streamlit

Stock price prediction is a challenging task that requires analyzing historical stock data and identifying patterns or trends to make informed predictions about future stock prices. In this technical documentation, we will explore the use of Long Short-Term Memory (LSTM) neural networks to predict stock prices. We will utilize the historical stock price data of Apple Inc. (AAPL) obtained from the Quandl API.

Data Retrieval and Preprocessing:
We begin by importing the necessary libraries such as pandas, numpy, matplotlib, and quandl. We use the Quandl API to fetch the historical stock price data for AAPL. The data is stored in a pandas DataFrame, and irrelevant columns are dropped. The data is then split into training and testing sets.

Data Scaling:
To improve the performance of our LSTM model, we use MinMaxScaler to scale the training data to a range between 0 and 1. This ensures that all the input features are on a similar scale, which is beneficial for training neural networks.

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

# 1. Alzheimer Detection from Live Speech 
Youtube link :- https://www.youtube.com/watch?v=Nbx6qjv7dMM&ab_channel=Vanshika
Neurocare is an app that aims to detect and provide support for Alzheimer's disease through early detection and wellness features.Using a voice memo recorded by the patient, Neurocare's trained model classifies speech as either AD (Alzheimer's disease) or non-AD. This classification is based on lexical analysis of grammar and pause durations. The model calculates a mental state examination score, aiding in reaching a diagnosis.
