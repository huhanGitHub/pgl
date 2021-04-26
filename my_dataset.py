import numpy as np
import os
import pandas as pd
import pgl


def build_graphs(atoms_features, edges_nodes, edges_features):
    graphs = []
    for atoms_feature, edges_node, edges_feature in zip(atoms_features, edges_nodes,
                                                       edges_features):
        g = pgl.Graph(edges=edges_node,
                      num_nodes=len(atoms_feature),
                      node_feat={'nfeat': np.array(atoms_feature, dtype='float32')},
                      edge_feat={'efeat': np.array(edges_feature, dtype='float32')})
        graphs.append(g.tensor())

    return graphs


class CustomDataset(object):
    def __init__(self, path1, path2, normalize=True):
        self.path1 = path1
        self.path2 = path2
        if not os.path.exists(self.path1):
            raise ValueError('empty dataset')
        self._load_data(normalize)
        self._load_data2()
        self._make_pairs()

    def _load_data(self, normalize=True):
        from sklearn.preprocessing import StandardScaler
        import scipy.sparse as sp
        data = pd.read_csv(self.path1)
        tag = data['Tag']
        self.tag = tag
        smile = data['Smile']
        self.smile = smile
        atom_features = data['Atom Features'].tolist()
        for i in range(len(atom_features)):
            atom_features[i] = atom_features[i][2:-2].split('], [')
            for j in range(len(atom_features[i])):
                atom_features[i][j] = atom_features[i][j].split(',')
                atom_features[i][j] = [int(jj) for jj in atom_features[i][j]]

        self.atom_features = atom_features
        edges = data['Edges'].tolist()
        for i in range(len(edges)):
            edges[i] = edges[i][2:-2].split('], [')
            for j in range(len(edges[i])):
                edges[i][j] = edges[i][j].split(',')
                edges[i][j] = [float(jj) for jj in edges[i][j]]
        edges_features = []
        edges_nodes = []
        for i in edges:
            node_edges = []
            node_edges_feature = []
            for ii in i:
                node_edges.append(tuple(ii[0:2]))
                node_edges_feature.append(ii[2])
            edges_features.append(node_edges_feature)
            edges_nodes.append(node_edges)
        edges_features = np.array(edges_features)
        self.edges_features = edges_features
        self.edges_nodes = edges_nodes

        plqy = data['PLQY'].tolist()
        self.plqy = plqy
        type = data['Type'].tolist()
        self.type = type
        log_plqy = data['Log PLQY'].tolist()
        self.log_plqy = log_plqy
        num_class_set = set(log_plqy)
        self.num_class = len(num_class_set)
        self.num_class_set = num_class_set
        # print('there are %d classes of labels in this dataset' % self.num_class)
        self.length = len(smile)

    def _load_data2(self):
        data = pd.read_csv(self.path2)
        tag = data['Tag']
        smile = data['Smile']
        atom_features = data['Atom Features'].tolist()
        for i in range(len(atom_features)):
            atom_features[i] = atom_features[i][2:-2].split('], [')
            for j in range(len(atom_features[i])):
                atom_features[i][j] = atom_features[i][j].split(',')
                atom_features[i][j] = [int(jj) for jj in atom_features[i][j]]

        self.atom_features2 = atom_features
        edges = data['Edges'].tolist()
        for i in range(len(edges)):
            edges[i] = edges[i][2:-2].split('], [')
            for j in range(len(edges[i])):
                edges[i][j] = edges[i][j].split(',')
                edges[i][j] = [float(jj) for jj in edges[i][j]]
        edges_features = []
        edges_nodes = []
        for i in edges:
            node_edges = []
            node_edges_feature = []
            for ii in i:
                node_edges.append(tuple(ii[0:2]))
                node_edges_feature.append(ii[2])
            edges_features.append(node_edges_feature)
            edges_nodes.append(node_edges)
        edges_features = np.array(edges_features)
        self.edges_features2 = edges_features
        self.edges_nodes2 = edges_nodes

        plqy = data['PLQY'].tolist()
        self.plqy2 = plqy
        type = data['Type'].tolist()
        self.type2 = type

        self.length2 = len(smile)

    def __len__(self):
        return self.length

    def _make_pairs(self):
        graphs1 = build_graphs(self.atom_features, self.edges_nodes, self.edges_features)
        graphs2 = build_graphs(self.atom_features2, self.edges_nodes2, self.edges_features2)

        dataset_pairs = [[i, j, k] for i, j, k in zip(graphs1, graphs2, self.log_plqy)]
        self.dataset_pairs = dataset_pairs
        break_point2 = int(self.length * 0.9)
        self.test_pairs = self.dataset_pairs[break_point2:]
        self.train_val_pairs = self.dataset_pairs[:break_point2]

    def get_train_val_test_pairs(self):
        break_point1 = int(self.length * 0.8)
        # break_point2 = int(self.length * 0.9)
        np.random.shuffle(self.train_val_pairs)
        train_pairs = self.train_val_pairs[:break_point1]
        val_pairs = self.train_val_pairs[break_point1:]
        test_pairs = self.test_pairs
        return train_pairs, val_pairs, test_pairs

