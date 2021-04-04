import numpy as np
import sys
from keras.models import load_model

#only add the below code if you are using TF with a GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Loads the trained model from 'saved_model' and
# uses it to predict the next words from a
# specific sequence

model = load_model('src/saved_model', compile=True)
print('Loaded model from disk')

#fetches the variables to store in 'chars' and in 'vocab_len'
variables_list = open('src/variables.txt', encoding='utf-8').read().split('\n')
chars = variables_list[:-1]
vocab_len = int(variables_list[-1])

#create a dictionary that relates chars to numbers and vice versa
num_to_char = dict((i,c) for i,c in enumerate(chars))
char_to_num = dict((c, i) for i,c in enumerate(chars))

#take a sentence as input and convert the characters into numbers
input_sentence = 'Detta är ett test '.lower()
pattern = []
pattern.append([char_to_num[i] for i in input_sentence])
pattern = pattern[0]

generated_sentence = [' ']
#generate 100 characters using NN model based on 'pattern'
for i in range(1,101):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_len)

    prediction = model.predict(x, verbose=0)

    #if the previous character is a space, randomly pick the first letter of the next word
    if generated_sentence[i-1] == ' ':
        index = np.random.choice(len(prediction[0]), p=prediction[0])
    else:
        index = np.argmax(prediction)

    result = num_to_char[index]
    generated_sentence.append(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]

#print out all characters in sequence
print('Generated sequence: ')
print(''.join(generated_sentence))