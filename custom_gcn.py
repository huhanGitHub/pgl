import tqdm
import pgl
import paddle
import paddle.nn as nn
from pgl.utils.logger import log
import numpy as np
import time
import argparse
from paddle.optimizer import Adam


class GCN(nn.Layer):
    """Implement of GCN
    """
    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=1,
                 hidden_size=64,
                 dropout=0.5):
        super(GCN, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.gcns = nn.LayerList()
        for i in range(self.num_layers):
            if i == 0:
                self.gcns.append(
                    pgl.nn.GCNConv(
                        input_size,
                        self.hidden_size,
                        activation="relu",
                        norm=True))
            else:
                self.gcns.append(
                    pgl.nn.GCNConv(
                        self.hidden_size,
                        self.hidden_size,
                        activation="relu",
                        norm=True))
            self.gcns.append(nn.Dropout(self.dropout))
        # self.gcns.append(pgl.nn.GCNConv(self.hidden_size, self.num_class))
        self.output = nn.Linear(self.hidden_size, self.num_class)

    def forward(self, graph, feature):
        for m in self.gcns:
            if isinstance(m, nn.Dropout):
                feature = m(feature)
            else:
                feature = m(graph, feature)
        logits = self.output(feature)
        output = logits.numpy()
        output = np.mean(output, axis=0)
        output = paddle.to_tensor(output)
        return output


class MyModel(nn.Layer):
    """Implement of GCN
    """
    def __init__(self,
                 input_size,
                 num_class,
                 num_layers=1,
                 hidden_size=64,
                 dropout=0.5):
        super(MyModel, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.gcns = nn.LayerList()
        for i in range(self.num_layers):
            if i == 0:
                self.gcns.append(
                    pgl.nn.GCNConv(
                        input_size,
                        self.hidden_size,
                        activation="relu",
                        norm=True))
            else:
                self.gcns.append(
                    pgl.nn.GCNConv(
                        self.hidden_size,
                        self.hidden_size,
                        activation="relu",
                        norm=True))
            self.gcns.append(nn.Dropout(self.dropout))
        # self.gcns.append(pgl.nn.GCNConv(self.hidden_size, self.num_class))
        self.linear = nn.Linear(self.hidden_size, self.num_class)
        self.output = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, graph1, feature1, graph2, feature2):
        for m in self.gcns:
            if isinstance(m, nn.Dropout):
                feature1 = m(feature1)
            else:
                feature1 = m(graph1, feature1)

        for m in self.gcns:
            if isinstance(m, nn.Dropout):
                feature2 = m(feature2)
            else:
                feature2 = m(graph2, feature2)

        feature1 = paddle.unsqueeze(feature1, axis=0)
        feature1 = paddle.sum(feature1, axis=1)
        feature1 = paddle.fluid.layers.fc(input=feature1, size=self.hidden_size)

        feature2 = paddle.unsqueeze(feature2, axis=0)
        feature2 = paddle.sum(feature2, axis=1)
        feature2 = paddle.fluid.layers.fc(input=feature2, size=self.hidden_size)
        # feature2 = paddle.squeeze(feature2, axis=0)

        chemical_features = paddle.concat(x=[feature1, feature2], axis=0)
        chemical_features = paddle.reshape(chemical_features, [self.hidden_size * 2, 1])
        chemical_features = paddle.fluid.layers.fc(input=chemical_features, size=1)
        chemical_features = paddle.squeeze(chemical_features, axis=1)
        output = self.output(chemical_features)
        # output = nn.functional.softmax(output)
        return output


def build_graph():
    # define the number of nodes; we can use number to represent every node
    num_node = 10
    # add edges, we represent all edges as a list of tuple (src, dst)
    edge_list = [(2, 0), (2, 1), (3, 1),(4, 0), (5, 0),
             (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
             (7, 2), (7, 3), (8, 0), (9, 7)]

    # Each node can be represented by a d-dimensional feature vector, here for simple, the feature vectors are randomly generated.
    d = 16
    feature = np.random.randn(num_node, d).astype("float32")
    # each edge has it own weight
    edge_feature = np.random.randn(len(edge_list), 1).astype("float32")

    # create a graph
    g = pgl.Graph(edges=edge_list,
                  num_nodes=num_node,
                  node_feat={'nfeat': feature},
                  edge_feat={'efeat': edge_feature})

    return g


def main_demo():
    gs = []
    for i in range(10):
        g = build_graph().tensor()
        gs.append(g)

    gs2 = []
    for i in range(10):
        g = build_graph().tensor()
        gs2.append(g)
    # g = build_graph()
    # print('There are %d nodes in the graph.' % g.num_nodes)
    # print('There are %d edges in the graph.' % g.num_edges)
    y = [0, 1, 1, 1, 0, 0, 0, 1, 0, 1]
    label = np.array(y, dtype="float32")
    # g = g.tensor()
    y = paddle.to_tensor(y)
    #gcn = GCN(16, 2)
    model = MyModel(16, 2)
    criterion = paddle.nn.loss.CrossEntropyLoss()
    optim = Adam(learning_rate=0.01,
                 parameters=model.parameters())
    model.train()
    for epoch in range(30):
        pred = []
        for g, g2 in zip(gs, gs2):
            pred.append(model(g, g.node_feat['nfeat'], g2, g2.node_feat['nfeat']))
        # node_logits = gcn(g, g.node_feat['nfeat'])
        # loss = criterion(node_logits, y)
        pred = paddle.to_tensor([i.numpy() for i in pred])
        loss = criterion(pred, y)
        loss.backward()
        optim.step()
        optim.clear_grad()
        print("epoch: %s | loss: %.4f" % (epoch, loss.numpy()[0]))


if __name__ == '__main__':
    main_demo()