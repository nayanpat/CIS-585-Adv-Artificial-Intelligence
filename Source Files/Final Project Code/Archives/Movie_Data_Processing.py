import keras
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.python.keras.layers import TextVectorization
from patsy import desc

'''make a list of descriptions'''
text = desc['Description'].tolist()

max_vocab_length = 20000 #expected number of words in the whole descriptions
max_length=25   #length of vector

'''define a vectorization layer'''
text_vectorizer = tf.python.keras.layers.TextVectorization(max_tokens = max_vocab_length, #how many words in the vocabulory
                                   standardize="lower_and_strip_punctuation",
                                   split="whitespace",
                                   ngrams=None ,# create group of words
                                   output_mode="int",
                                   output_sequence_length=max_length) #length of sequence

text_vectorizer.adapt(text)  #adapt/fit the model with text data
text_vector = text_vectorizer(text)  #convert the text to vector
text_vector[:2]  #check first two vectors

