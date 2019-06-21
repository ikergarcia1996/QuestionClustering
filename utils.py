from six import text_type as unicode
from six import string_types
from sklearn.preprocessing import normalize
import numpy as np
import datetime
from heapq import nlargest
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


# [ES] Entrada: String. Salida: String convertido al formato unicode
def word_to_unicode(w):
    if isinstance(w, string_types) and not isinstance(w, unicode):
        return unicode(w, encoding="utf-8")
    else:
        return w


# [ES]: ws: Lista de palabras. unicode: Transforma las palabras al formato unicode. lower: Transforma las palabras a minúsculas
def standardize_words(ws, to_unicode=False, lower=False):
    for i in range(len(ws)):
        if to_unicode:
            ws[i] = word_to_unicode(ws[i])
        if lower:
            ws[i] = ws[i].lower()
    return ws


def isInt_float(s):
    try:
        return float(str(s)).is_integer()
    except:
        return False


def has_header(path):
    with open(path) as f:
        first_line = f.readline()
        first_line = first_line.split(' ')
        if len(first_line) == 2:
            return 1
        return 0


def get_dimensions(path):
    with open(path) as f:
        first_line = f.readline()
        first_line = first_line.split(' ')
        if len(first_line) == 2:
            return int(first_line[1])
        else:
            return None


def get_num_words(path):
    with open(path) as f:
        first_line = f.readline()
        first_line = first_line.split(' ')
        if len(first_line) == 2:
            return int(first_line[0])
        else:
            return None


def vocab_from_path(path):
    # Dado un embedding (path) devuelve su vocabulario. Set
    words = []

    with open(path) as f:
        dims = get_dimensions(path)

        if dims:
            next(f)

            for line in f:
                l = line.split()
                wi = len(l) - dims
                st = ''.join(l[0:wi])
                words.append(st)
        else:
            for line in f:
                words.append(line.split()[0])

    return set(words)


def normalize_vector(vector, norm='l2'):
    vector = np.asarray(vector).reshape(-1, len(vector))
    return normalize(vector, norm=norm)[0]


def printTrace(message):
    print("<" + str(datetime.datetime.now()) + ">  " + str(message))


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def get_largest_index(list_cos,k):
    n = nlargest(k, enumerate(list_cos), key=lambda x: x[1])
    return [x[0] for x in n]


def generate_dictionary_for_vecmap(path1, path2):
    vocab1 = vocab_from_path(path1)
    vocab2 = vocab_from_path(path2)
    return set.intersection(vocab1, vocab2)


def print_dictionary_for_vecmap(filename, dict):
    with open(filename, 'w') as file:
        for w in dict:
            if w != '' and w != '\n':
                print(w + ' ' + w, file=file)


def length_normalize(vectors):
    norms = np.sqrt(np.sum(vectors ** 2, axis=1))
    norms[norms == 0] = 1
    return vectors / norms[:, np.newaxis]


def normalize_questions(question):
    #question = question.lower()
    question = re.sub(r"[^a-zA-Z0-9]+", r" ", question)
    question = word_tokenize(question)
    question = [w for w in question if not w in set(stopwords.words('spanish'))]
    return question


def calculate_cosine_simil(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def perf_measure(y_actual, y_hat):
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return TP, FP, TN, FN