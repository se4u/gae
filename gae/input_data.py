import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
    x, tx, allx, graph = tuple(objects)
    # """
    # x = <140x1433 sparse matrix of type '<type 'numpy.float32'>'
    #    with 2647 stored elements in Compressed Sparse Row format>
    # tx = <1000x1433 sparse matrix of type '<type 'numpy.float32'>'
    #    with 17955 stored elements in Compressed Sparse Row format>
    # allx = <1708x1433 sparse matrix of type '<type 'numpy.float32'>'
    #    with 31261 stored elements in Compressed Sparse Row format>
    # type(graph) = <type 'collections.defaultdict'>
    # """
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # """
    # (Pdb) p features
    #     <2708x1433 sparse matrix of type '<type 'numpy.float32'>'
    #     with 49216 stored elements in LInked List format>
    # (Pdb) p adj
    #     <2708x2708 sparse matrix of type '<type 'numpy.int64'>'
    #     with 10556 stored elements in Compressed Sparse Row format>
    # (Pdb) p adj.data.max()
    # 1
    # (Pdb) p adj.data.min()
    # 1
    # (Pdb) p features.data.max()
    # [1.0, 1.0, 1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0, 1.0, 1.0] (len=30)
    # (Pdb) p features.data.min()
    # [1.0]
    # """
    return adj, features
