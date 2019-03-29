from six import iteritems
from utils import standardize_words


class Vocabulary(object):
    word_id = []
    id_word = []

    # [ES] words: Lista de palabras. to_unicode: Transforma las palabras a formato unicode. lower: Transforma todas la palabras a minúsculas
    def __init__(self, words=[], to_unicode=True, lower=False):
        words = standardize_words(words, to_unicode, lower)
        self.word_id = {w: i for i, w in enumerate(words)}
        self.id_word = {i: w for w, i in iteritems(self.word_id)}

    def __len__(self):
        return len(self.word_id)

    @property
    def words(self):
        return list(self.word_id)

        # [ES]: Dado un indice i (natural) devuelve la palabra número i del vocabulario. En caso de que no exista eleva una exepción

    def index_to_word(self, word):
        try:
            return self.id_word[word]
        except KeyError as err:
            raise

    # [ES]: Dada una palabra (String) devuelve la posición (natural), que ocupa en el vocabulario. En caso de que no exista eleva una exepción
    def word_to_index(self, word):
        try:
            return self.word_id[word]
        except KeyError as err:
            raise
