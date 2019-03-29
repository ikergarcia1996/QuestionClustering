from six import text_type as unicode
from six import string_types
from sklearn.preprocessing import normalize
import numpy as np
import datetime
from heapq import nlargest

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

