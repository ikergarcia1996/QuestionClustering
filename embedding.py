import numpy as np
import regex
from sklearn.preprocessing import normalize
from vocabulary import Vocabulary
import pandas as pd
import logging
import io
import codecs
from utils import get_dimensions

class Embedding(object):
    vocabulary = Vocabulary()
    vectors = []

    # [ES] vocabulary: Objeto de la clase Vocabulary que contiene las palabras del embedding. vector: Matriz de vectores, Para i de 0 a n, la fila i se corresponderá con la palabra i del vocabulario.
    # En caso de existir palabras duplicadas se elevará una excepción.
    def __init__(self, vocabulary=Vocabulary(), vectors=[]):

        self.vocabulary = vocabulary
        self.vectors = np.array(vectors)

        # [EN]: Check if we have the same number of words and vectors
        # [ES]: Comprobar si tenemos el mismo numero de vectores que de palabras


        if len(self.vocabulary) != self.vectors.shape[0]:
            raise ValueError("We have a different number of words and vectors. We have {} words and {} vectors".format(
                len(self.vocabulary), self.vectors.shape[0]))

        # [EN]: Test if there are duplicated words
        # [ES]: Comprobar si tenemos palabras duplicadas
        if len(self.vocabulary) != len(set(self.vocabulary.words)):
            logging.warning("Vocabulary has duplicates")

    def __len__(self):
        return len(self.vocabulary)

    @property
    def words(self):
        return self.vocabulary.words

    @property
    def dims(self):
        return self.vectors.shape[1]

    # [ES] Dada una palabra (String), devuelve su vector (Array de floats). En caso de que la palabra no existe eleva una excepción
    def word_to_vector(self, word, lower=False):
        try:
            if lower:
                return self.vectors[self.vocabulary.word_to_index(word.lower())]
            else:
                return self.vectors[self.vocabulary.word_to_index(word)]
        except KeyError as err:
            raise

    def word_to_index(self, word):
        try:
            return self.vocabulary.word_to_index(word)
        except KeyError as err:
            raise

    def list_to_index(self, words):
        try:
            indexes = []
            for w in words:
                indexes.append(self.vocabulary.word_to_index(w))
            return indexes

        except KeyError as err:
            raise

    def words_to_matrix(self, vocab):
        try:
            return [self.word_to_vector(x) for x in vocab]
        except KeyError as err:
                raise


    # [ES] Normalización L2 por filas. Para cada vector asegura que la raiz cuadrada de la suma de los cuadros es igual a 1.
    # [ES] Si replace==True se sustituarian los vectores actuales por los vectores normalizados. En caso contrario, se devolverá un nuevo embedding con las mismas palabras que el actual pero los vectores normalizados.
    def length_normalize(self, replace=True):
        norms = np.sqrt(np.sum(self.vectors ** 2, axis=1))
        norms[norms == 0] = 1
        if replace:
            self.vectors = self.vectors / norms[:, np.newaxis]

        else:
            return Embedding(vectors=self.vectors / norms[:, np.newaxis], vocabulary=self.vocabulary)

    # [ES] Normalización L2 por columnas. . Para cada columna asegura que la raiz cuadrada de la suma de los cuadros es igual a 1.
    # [ES] Si replace==True se sustituarian los vectores actuales por los vectores normalizados. En caso contrario, se devolverá un nuevo embedding con las mismas palabras que el actual pero los vectores normalizados.
    def length_normalize_dimensionwise(self, replace=True):
        norms = np.sqrt(np.sum(self.vectors ** 2, axis=0))
        norms[norms == 0] = 1
        if replace:
            self.vectors = self.vectors / norms

        else:
            return Embedding(vectors=self.vectors / norms, vocabulary=self.vocabulary)




# [ES] Carga un embedding desde un directorio, devuelve un objeto del tipo embedding.
# [ES] Parámetros:
# path: Ruta (obligatorio)
# format: text, archivo de texto, con o sin encabezado. bin archivo binario (w2v). senna modo especial para estos embeddings. vgg modo especial para los embedding de imágenes. DT_embedddings: Modo espacial para leer estos embeddings
# vocabulary: Si es None cargará todas las palabras del fichero, si es una lista de palabras [cat, dog...] solo se cargarán las palabras pertenecientes a esa lista, útil si solo se quiere evaluar los embeddings en unos datasets
# length_normalize: Normalización L2 por filas de los embeddings cargados
# normalize_dimensionwise: Normalización L2 por columanas de los embeddigs cargados.
# En caso de que length_normalize y normalize_dimensionwise sean TRUE. Primero se realiza la normalización por columnas y después por filas.
# to_unicode: Transforma a unicode las palabras del embedding
# lower: Transforma a minúsculas las palabras del emnedding
# path2: En caso de que el formato sea "senna" o "DT_embedings" en path irá el fichero con las palabras, y en path2 el fichero con los vectores
# dims_restricton: Entero i (dimensiones del embedding). Por cada fila leida se tomará como número los elementos n..n-i y los elementos 0..n-i-1 se concaterán y serán tomados como palabra. Muy útil para embeddings con palabras que contienen espacio o carácteres extraños que son tomados como espacios.
# delete_duplicates: Elimina los duplicados del embeddig, por cada palabra duplicada se mantendrá solo la primera que aparezca en el fichero. Solo recomendable para embeddings con mucha suciedad como los obtenidos de common crawl, donde algunas palabras con caracteres extraños no reconocidos por python se leen como si fueran la misma, eliminar estas palabras no tiene impacto alguno en el embedding, por ejemplo en el caso de FastText CC solo se eliminan 4 palabras.
# method_vgg: En caso de que el formato sea "vgg", si este atributo es "delete" se eliminarán los duplicados de la misma forma explicada en el atributo anterior. En caso de que sea "average" se hará la media de los vectores de las palabras duplicadas. Este es un embedding que tiene la peculiaridad de tener un gran número de duplicados

def load_embedding(path, format="text", vocabulary=None, length_normalize=True, normalize_dimensionwise=False,
                   to_unicode=True, lower=False, path2='', delete_duplicates=False,
                   method_vgg="delete"):
    assert format in ["text", "bin", "senna", "vgg", "DT_embeddings"], "Unrecognized format"

    if vocabulary is not None:
        if len(set(vocabulary)) != len(vocabulary):
            logging.warning(
                "Provided vocabulary has duplicates. IMPORTANT NOTE: The embedding that this function will return will not have duplicates.")



    if format == "text":
        dims_restriction = get_dimensions(path)
        vocab, matrix = from_TXT(path, vocabulary, dims_restriction)

    if format == "bin":
        vocab, matrix = from_BIN(path, vocabulary)

    if format == "senna":
        vocab, matrix = from_SENNA(path, path2)

    if format == "vgg":
        vocab, matrix = from_vgg(path, method_vgg)

    if format == "DT_embeddings":
        vocab, matrix = from_DT_embeddings(path, path2)

    if delete_duplicates:
        remove_duplicates(vocab, matrix)

    vocabulary = Vocabulary(vocab, to_unicode, lower)
    e = Embedding(vocabulary=vocabulary, vectors=matrix)

    if normalize_dimensionwise:
        e.length_normalize_dimensionwise()

    if length_normalize:
        e.length_normalize()

    return e



def remove_duplicates(words, vectors):
    seen = set()
    seen_add = seen.add
    duplicate_indexes = [idx for idx, item in enumerate(words) if item in seen or seen_add(item)]
    for d in reversed(duplicate_indexes):
        logging.warning("Word {} deleted".format((words[d])))

        del (words[d])
        del (vectors[d])



def from_TXT(path, vocabulary=None, dims_restriction=None):
    words = []
    vectors = []

    # with codecs.open(path, "r",encoding='utf-8', errors='ignore') as f:
    with open(path) as f:
        if dims_restriction:
            next(f)
        for line_no, line in enumerate(f):
            l = line.split()

            if vocabulary is None:
                try:
                    if not dims_restriction:
                        vectors.append(np.asarray(l[1:]).astype(np.float32))
                        words.append(l[0])
                    else:
                        wi = len(l) - dims_restriction
                        st = ''.join(l[0:wi])
                        words.append(st)
                        vectors.append(np.asarray(l[wi:]).astype(np.float32))

                except ValueError:
                    logging.warning(
                        "Line {}.Error reading the vector for the word {}... Word has been omitted".format(line_no,
                                                                                                           l[0:3]))

            elif l[0] in vocabulary:
                try:
                    if not dims_restriction:
                        vectors.append(np.asarray(l[1:]).astype(np.float32))
                        words.append(l[0])
                    else:
                        wi = len(l) - dims_restriction
                        st = ''.join(l[0:wi])
                        words.append(st)
                        vectors.append(np.asarray(l[wi:]).astype(np.float32))
                except ValueError:
                    logging.warning(
                        "Line {}.Error reading the vector for the word {}... Word has been omitted".format(line_no,
                                                                                                           l[0:3]))

    return words, vectors


def from_BIN(path, vocabulary=None):
    with io.open(path, 'rb') as file:
        words = []

        header = file.readline()
        vocab_size, layer1_size = list(map(int, header.split()))
        vocab_size_aux = vocab_size

        if vocabulary is None:
            vectors = np.zeros((vocab_size, layer1_size), dtype=np.float32)
        else:
            vectors = []

        binary_len = np.dtype("float32").itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = file.read(1)
                if ch == b' ':
                    break
                if ch != b'\n':
                    word.append(ch)

            if vocabulary is None:

                words.append(b''.join(word).decode("latin-1"))

                vectors[line, :] = np.fromstring(file.read(binary_len), dtype=np.float32)

            else:
                w = b''.join(word).decode("latin-1")
                if w in vocabulary:
                    words.append(w)
                    vectors.append(np.fromstring(file.read(binary_len), dtype=np.float32))

                else:
                    vocab_size_aux -= 1
                    file.read(binary_len)

        if len(words) != vocab_size_aux:
            raise ValueError("Header says that there are {} words, but {} were read".format(vocab_size_aux, len(words)))

        return words, vectors


def from_SENNA(path_words, path_vectors):
    words = []
    vectors = []

    with open(path_words) as f:
        for line in f:
            l = line.split()
            words.append(l[0])

    with open(path_vectors) as f:
        for line in f:
            l = line.split()
            vectors.append(np.asarray(l).astype(np.float32))
    return words, vectors


def from_vgg(path, method_vgg="delete"):
    data = pd.read_csv(path, sep=',', header=None)
    words = []
    vectors = []
    for i in range(len(data.index)):
        w = data.iloc[i, 0]
        if w not in words:
            words.append(w)
            vectors.append(np.array(data.iloc[i, 1:]).astype(np.float32))
        else:
            if method_vgg == "mean":
                ind = words.index(w)
                vectors[ind] = (vectors[ind] + np.array(data.iloc[i, 1:]).astype(np.float32)) / 2

    return words, vectors


def from_DT_embeddings(path_nodes, path_vectors):
    vectors = []
    nodes = {}
    words = []
    with codecs.open(path_nodes, "r", encoding='utf-8', errors='ignore') as fnodes:
        for line_no, line in enumerate(fnodes):
            wt, wn = line.split()
            w = wt.split('/')[0]
            nodes.update({wn: w})

    with codecs.open(path_vectors, "r", encoding='utf-8', errors='ignore') as fwords:
        next(fwords)
        for line_no, line in enumerate(fwords):
            l = line.split()
            vectors.append(np.asarray(l[1:]).astype(np.float32))
            words.append(nodes[l[0]])

    return words, vectors

