#!/usr/bin/python

"""

TODO
maybe use sparse arrays

TODO
used 1000-weight to choose maximum, not minimum
1000 is to enforce positive weights (maybe not necessary)

TODO
make sure attachment score is correct

TODO
add the distance feature
"""



import networkx
import numpy
import edmonds
import alternative_edmonds
from nltk.corpus import dependency_treebank
import features
import logging
from collections import namedtuple

# for debugging
import pdb
import send_mail
import pickle

# the learning ratio
ETA = 1
# number of iterations through all training set
N_ITERATIONS = 2

Arc = namedtuple('Arc', ('head', 'weight', 'tail'))  # reversed arc


all_trees = dependency_treebank.parsed_sents()
train_data = all_trees[:int(0.9 * len(all_trees))]
test_data = all_trees[int(0.9 * len(all_trees)):]
# TODO REMOVE THIS
train_data = all_trees[:int(0.1 * len(all_trees))]

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.handlers:
    l = logging.StreamHandler()
    l.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(l)


def make_graph(s, theta, f):
    """
    :param s: list word (word,tag)
    :param theta: the weights vector
    :param f: feature function
    :return: graph with weights on the edges
    """
    G = dict()
    s_len = len(s)
    for i in xrange(s_len):
        G[i] = dict()
        # j>=1 since no edges enter root
        for j in xrange(1, s_len):
            if i != j:
                G[i][j] = numpy.dot(theta, f(s, i, j))
    return G


def graph_to_arc_list(G):
    arc_list = []
    for tail, heads_and_weights in G.items():
        for head, weight in heads_and_weights.items():
            arc_list.append(Arc(tail, weight, head))
    return arc_list


def arc_list_to_graph(arc_dict):
    G = dict()
    for key, arc in arc_dict.items():
        if arc.head not in G:
            G[arc.head] = dict()
        G[arc.head][arc.tail] = arc.weight
    return G


def get_words_and_tags(tree):
    """
    :param tree:
    :return: a list of (word,tag)
    """
    assert tree.nodes[0]['word'] is None
    s = []
    for i, val in tree.nodes.items():
        s.append((val['word'], val['tag']))
    return s


def simplify_tree(tree):
    G = dict()
    for index, node in tree.nodes.items():
        if len(node['deps']) == 0:
            G[index] = dict()
        else:
            G[index] = {i:0 for i in node['deps'].values()[0]}
    return G


def get_sum_of_features_on_arcs(tree, sent, f):
    if not isinstance(tree, dict):
        tree = simplify_tree(tree)
    n_features = f(0, 0, 0, True)
    s = numpy.array([0] * n_features)
    for edge in all_edges(tree):
        s += f(sent, edge[0], edge[1])
    return s

# TODO remove depth constrain
def all_edges(tree, index=0, depth=0):
    """
    return a list of all edges of a tree
    """
    if depth == 30:
        pdb.set_trace()
    if index not in tree or len(tree[index]) == 0:
        # for a leaf empty list
        logger.debug("leaf: %d" % index)
        return []
    edges = []
    for child in tree[index].keys():
        logger.debug("edge %d,%d" % (index, child))
        edges.append((index, child))
        # recurse over of sub-tree
        edges.extend(all_edges(tree, child, depth + 1))
    return edges

#
# def max_st(G, debug=False):
#     """
#     find the MAXIMUM spanning tree
#     :param G:
#     :return: MST
#     """
#     Gnx = networkx.DiGraph()
#     for i in xrange(len(G)):
#         Gnx.add_node(i)
#     for i in G:
#         for j in G[i]:
#             Gnx.add_edge(i, j, weight=-G[i][j])
#     if debug:
#         return Gnx
#     mst = networkx.algorithms.tree.Edmonds(Gnx, 0).find_optimum()
#
#     return mst

def max_st(G, debug=False):
    """
    find the MAXIMUM spanning tree
    :param G:
    :return: MST
    """
    new_G = {}
    for i in G:
        new_G[i] = {}
        for j in G[i]:
            new_G[i][j] = -1 * G[i][j]
    #mst = edmonds.mst(0, new_G)
    mst = arc_list_to_graph(alternative_edmonds.min_spanning_arborescence(graph_to_arc_list(new_G), 0))
    return mst


def make_f(train_data):
    words = set([])
    tags = set([])
    for tree in train_data:
        for w, t in get_words_and_tags(tree):
            words.add(w)
            tags.add(t)
    logger.info("Making Feature Function (%d words, %d tags)"
                % (len(words), len(tags)))
    return features.make_feature_function(words, tags)


def perceptron(train_data):
    f = make_f(train_data)
    n_features = f(0, 0, 0, True)  # magic input to get the number of features
    logger.info("Doing the stuff")
    sum_thetas = numpy.array([0] * n_features)
    theta = numpy.array([0] * n_features)
    for n in xrange(N_ITERATIONS):
        for i, tree in enumerate(train_data):
            logger.info('Working on tree # %d' % i)
            sent = get_words_and_tags(tree)
            t_prime = choose_dependency_tree(sent, theta, f)

            sum_over_t = get_sum_of_features_on_arcs(tree, sent, f)
            sum_over_t_prime = get_sum_of_features_on_arcs(t_prime, sent, f)
            theta = theta + ETA * (sum_over_t - sum_over_t_prime)
            sum_thetas += theta

    return sum_thetas / (1. * N_ITERATIONS * len(train_data))


def attachment_score_on_test(f, theta, test_data):
    sum_attachment = 0
    for t in test_data:
        sent = get_words_and_tags(t)
        our_t = choose_dependency_tree(sent, theta, f)
        sum_attachment += attachment_score(our_t, t)
    return sum_attachment / len(test_data)


def choose_dependency_tree(sent, theta, f):
    """
    choose the best dependency tree
    :param sent: list of (word,tag)
    :param theta: the weights vector
    :param f: the feature function
    :return: the tree
    """
    G = make_graph(sent, theta, f)
    tree = max_st(G)
    assert len(all_edges(tree, 0)) == len(sent) - 1
    return tree


def attachment_score(t, t_gold_standard):
    t_gold_standard = simplify_tree(t_gold_standard)
    edges = set(all_edges(t))
    gold_edges = set(all_edges(t_gold_standard))
    n_words = len(t)
    logger.debug("correct edges: " + str(edges & gold_edges))
    return len(edges & gold_edges) / (1. * n_words)

if __name__ == "__main__":
    theta = perceptron(train_data[:5])
    pickle.dump(theta, open('theta.pickle', 'w'))
    send_mail.send_mail('Perceptron is done', 'Do not reply to this mail. Click here to unsubscribe')
