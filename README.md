# NLP
# 1. Neural Machine Translation 
Link :- https://github.com/vanshika230/LGMVIP_DataScienceIntern/blob/main/NMT.ipynb

Implemented Sequential LSTM to translate dialogues from English to Hindi language, achieving accuracy of 89%.

Leveraged analytical techniques to handle pre-processing datasets and implemented regularization techniques
increasing performance by 8%

--> About the project :- 

The code imports the necessary libraries and modules for data processing and modeling.

It reads a CSV file containing English and Hindi sentence pairs using the pandas library.

Some initial data exploration and cleaning operations are performed, such as filtering data from a specific source, dropping columns, removing null values, converting text to lowercase, and applying regular expressions for text cleaning.

The code calculates the maximum sequence length for both English and Hindi sentences.

Unique English and Hindi words are extracted and stored in separate lists.

Tokenizers are created using Keras to convert the sentences into sequences of integers.

The sequences are generated for both English and Hindi sentences using the tokenizers.

The word indices for Hindi words are obtained from the tokenizer.

Model Architecture

The NMT model follows a sequential architecture, which incorporates several essential layers to handle text translation. Here are the key components of the model:

Embedding Layer: This layer plays a crucial role in mapping the input words to dense vectors of a fixed size. By representing words in a continuous vector space, it captures semantic relationships, enabling better translation performance.

LSTM (Long Short-Term Memory) Layer: LSTM is a type of recurrent neural network layer that excels at capturing long-range dependencies in the input sequence. It processes the input word vectors and learns to encode contextual information, improving translation accuracy.

RepeatVector Layer: To match the desired output sequence length, this layer repeats the output of the previous LSTM layer multiple times. This repetition enables the model to generate a sequence of words based on the input, facilitating the translation process.

LSTM Layer (with return_sequences=True): Similar to the previous LSTM layer, this layer is configured to return sequences instead of a single output. It aids in decoding the input representation and generating the output sequence, contributing to the accurate translation of the text.

Model Saving
Once the model is trained, you can save it for future use. You can accomplish this by using the save() method, which saves the entire model to a file. To save the model, simply call model.save('NMTmodel'). 


Dense Layer: The final dense layer applies a softmax activation function to produce a probability distribution over the Hindi vocabulary. It selects the most likely Hindi word for each position in the output sequence, ensuring the generated translations are linguistically appropriate.
