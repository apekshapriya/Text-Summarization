import warnings
warnings.filterwarnings("ignore")

import os, re, string
import numpy as np, pickle, pandas as pd
import gensim
from sklearn.model_selection import train_test_split
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
port = PorterStemmer()
lmtzr = WordNetLemmatizer()

import matplotlib.pyplot as plt
english_dict_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/english_words.txt"))
with open(english_dict_path) as word_file:
    english_words = set(word.strip().lower() for word in word_file)


def csv_load_data(file_path):
    """
    loads csv data nto a dataframe
    :param file_path:
    :return:  input_text, labels
    """

    data = pd.read_csv(file_path)

    return data['text'], data['headlines']

def remove_non_english_words(sentence):
    """
    remove words which are not in english dictionary
    :param sentence:
    :return: sentence
    """
    return " ".join(str(w) for w in sentence.split() if w in english_words)


def remove_punc_alternative_with_space(sentence):
    """
    Remove punctuations from the sentence and replaces it with a space
    @param sentence:
    @return: sentence
    """
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub(' ', sentence)


def remove_extra_space(sentence):
    """
    removes extra space between words
    :param sentence:
    :return: sentence
    """
    return " ".join(str(w) for w in sentence.split() if len(w) > 0)


def preprocess_data(input_text, labels):
    """

    :param input_text:
    :param labels:
    :return:
    """
    input_text_list, labels_text_list = [],[]

    for index, sentence in enumerate(input_text):
        input_text_list.append(remove_extra_space(remove_non_english_words(remove_punc_alternative_with_space(sentence))))

    for index, sentence in enumerate(labels):
        labels_text_list.append(remove_extra_space(remove_non_english_words(remove_punc_alternative_with_space(sentence))))

    for index, sentence in enumerate(input_text_list):
        input_text_list[index] = sentence.split()

    for index, sentence in enumerate(labels_text_list):
        labels_text_list[index] = sentence.split()

    total_num_words = ([len(one_sentence) for one_sentence in input_text_list])

    # plt.hist(total_num_words, bins=np.arange(0, 60, 2))
    # plt.show()
    # by plotting, we come to know that on avg max length of the sentences are 45
    max_length = 45
    # So we take each sentence upto 45 words only
    for index, text in enumerate(input_text_list):
        input_text_list[index] = text[:max_length]

    for index, text in enumerate(labels_text_list):
        labels_text_list[index] = text[:max_length]

    return input_text_list, labels_text_list, max_length


def create_dictionary(input_text, labels):
    """

    :param input_text:
    :param labels:
    :return: vocab
    """
    vocab = []
    for sentence in input_text:
        vocab.extend(word for word in sentence)
    for sentence in labels:
        vocab.extend(word for word in sentence)

    return set(list(vocab))


def create_word_and_ids(vocab):
    """

    :param vocab:
    :return:
    """

    id_to_word = {k:v for k,v in enumerate(vocab)}
    word_to_id = {v:k for k,v in id_to_word.items()}

    return id_to_word, word_to_id


def create_word_to_index_sentences(word_to_id, input_text, labels):
    """
    convert each word in a sentence to its corresponding index, as words take more memory
    :param word_to_id:
    :param input_text:
    :param labels:
    :return:
    """

    for index, text in enumerate(input_text):
        input_text[index] = [word_to_id[word] for word in text]

    for index, text in enumerate(labels):
        labels[index] = [word_to_id[word] for word in text]



    return input_text, labels


def lemmatize(word):
    """
    Lemmatize each word of the sentence
    @param sentence:
    @return:
    """

    return lmtzr.lemmatize(str(word))

def stem(word):
    """
    Lemmatize each word of the sentence
    @param sentence:
    @return:
    """

    return port.stem(str(word))

def pad_sentence(input_text, labels, max_len):
    """
    padding each sentence with "unknown" to make them of equal length.
    :param max_len:
    :return:
    """
    for index, text in enumerate(input_text):
        input_text[index] = [input_text[index] +
                                       ['unk']*(max_len-len(input_text[index]))][0]

    for index, text in enumerate(labels):
        labels[index] = [labels[index] +
                                       ['unk']*(max_len-len(labels[index]))][0]
    return input_text, labels

def create_word_embeddings_for_data(vocab, id_to_word):
    """
    creating word embedding from the data using word_to_vec model of google
    :param vocab:
    :param id_to_word:
    :return:
    """
    word_vec = gensim.models.KeyedVectors.load_word2vec_format\
        ('../data/GoogleNews-vectors-negative300.bin', binary=True)

    embedding = np.zeros([len(vocab), 300])
    for key, value in id_to_word.iteritems():

        try:
            embedding[key] = word_vec[value]

        except:
            value_lemmatized = lemmatize(value)
            value_stemmed = stem(value_lemmatized)

            if value_lemmatized in word_vec:
                embedding[key] = word_vec[value_lemmatized.encode('ascii', 'ignore')]

            elif value_stemmed in word_vec:
                embedding[key] = word_vec[value_stemmed.encode('ascii', 'ignore')]

            elif len(wordnet.synsets(value_lemmatized)) > 0:

                syn = wordnet.synsets(value_lemmatized)[0]
                lemma_name = [lemma.name() for lemma in syn.lemmas()]

                for name in lemma_name:
                    if name not in word_vec:

                        continue
                    else:

                        embedding[key] = word_vec[name]
            else:
                print value
    embedding[word_to_id['unk']] = np.zeros(300)
    return embedding


def split_train_val_test(input_text, labels):

    idx = np.random.permutation(len(labels))


    input_text = np.array(input_text)
    labels = np.array(labels)
    input_text, labels = input_text[idx], labels[idx]

    X_train, X_test, y_train, y_test = train_test_split(input_text, labels,

                                                        test_size=0.25)

    X_val, y_val = X_test[:100], y_test[:100]
    X_test, y_test = X_test[100:], y_test[100:]
    return X_train, X_test, X_val, y_train, y_test, y_val


def save_data(input_text, labels, embedding):

    """
    saves input, labels, embedding
    :param input_text:
    :param labels:
    :param embedding:
    :return:
    """

    X_train, X_test, X_val, y_train, y_test, y_val = split_train_val_test(input_text, labels)

    with open("../processed_data/X_train.pkl", "wb") as inp:
        pickle.dump(X_train, inp)

    with open("../processed_data/X_test.pkl", "wb") as lab:
        pickle.dump(X_test, lab)

    with open("../processed_data/X_val.pkl", "wb") as lab:
        pickle.dump(X_val, lab)

    with open("../processed_data/y_train.pkl", "wb") as lab:
        pickle.dump(y_train, lab)

    with open("../processed_data/y_test.pkl", "wb") as lab:
        pickle.dump(y_test, lab)

    with open("../processed_data/y_val.pkl", "wb") as lab:
        pickle.dump(y_val, lab)

    with open("../processed_data/embedding_vector.pkl", "wb") as embd:
        pickle.dump(embedding, embd)


if __name__ == "__main__":

    input_text, summarized_text = csv_load_data("../data/news_summary.csv")
    input_text, summarized_text, max_length = preprocess_data(input_text, summarized_text)
    input_text, summarized_text = pad_sentence(input_text, summarized_text, max_length)
    vocab = create_dictionary(input_text, summarized_text)
    id_to_word, word_to_id = create_word_and_ids(vocab)
    input_text, labels = create_word_to_index_sentences(word_to_id, input_text, summarized_text)
    embedding = create_word_embeddings_for_data(vocab, id_to_word)
    split_train_val_test(input_text, summarized_text)
    save_data(input_text, summarized_text, embedding)

