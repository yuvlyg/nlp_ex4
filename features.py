import itertools
from scipy.sparse import lil_matrix


def n_choose_2(n):
    return n * (n-1) / 2


def make_empty_vector(n):
    return lil_matrix((1, n), dtype='i')


def make_feature_function(all_words, all_tags):
    n_features = len(all_words)**2 + len(all_tags)**2
    word_pair_to_index = {}
    tag_pair_to_index = {}
    index = 0
    for w1,w2 in itertools.product(all_words, all_words):
        word_pair_to_index[(w1,w2)] = index
        index += 1
    for t1,t2 in itertools.product(all_tags, all_tags):
        tag_pair_to_index[(t1,t2)] = index
        index += 1

    def f(s, i, j, return_len=False):
        if return_len:
            return n_features
        res = make_empty_vector(n_features)

        # elements of s[i] are the word-tag tuples,
        w1,w2 = s[i][0], s[j][0]
        t1,t2 = s[i][1], s[j][1]

        # turn on the corresponding feature
        wp_index = word_pair_to_index[(w1, w2)]
        res[(0, wp_index)] = 1
        tp_index = tag_pair_to_index[(t1, t2)]
        res[(0, tp_index)] = 1
        return res

    return f

