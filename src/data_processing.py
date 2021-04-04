import numpy
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# Reads in the 'data.txt' comments and firstly tokenizes
# input. Then creates the training and test data sequences
# and lastly saves some variables to 'variables.txt'

#nltk.download('stopwords')
file = open('src/data.txt', encoding='utf-8').read()

#transform comments into a vector of single words
def tokenize_words(file):
    file = file.lower()

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(file)

    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return ' '.join(filtered)

processed_file = tokenize_words(file)
chars = sorted(list(set(processed_file)))
char_to_num = dict((c, i) for i,c in enumerate(chars))

file_len = len(processed_file)
vocab_len = len(chars)

seq_length = 100
x_data = []
y_data = []

#create training and label data
for i in range(0, file_len - seq_length, 1):
    in_seq = processed_file[i:i + seq_length]

    out_seq = processed_file[i + seq_length]

    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

#save variables in a separate file
with open('src/variables.txt', 'w', encoding='utf-8') as f:
    for character in chars:
        f.write("%s\n" % character)
    f.write(str(vocab_len))

print('Done processing data')

def get_chars():
    return chars

def get_x_data():
    return x_data

def get_y_data():
    return y_data

def get_vocab_len():
    return vocab_len

def get_seq_len():
    return seq_length