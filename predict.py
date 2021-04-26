import pandas as pd
from my_dataset import CustomDataset
from custom_gcn import MyModel
import paddle
import tqdm
import time
from paddle.optimizer import Adam
import pgl
import numpy as np
from train_main import eval, train, build_graphs


def load_train(model_path, adam_path):
    batch_size = 64
    num_classes = 4
    depth = 44

    # load
    model_state_dict = paddle.load(model_path)
    adam_state_dict = paddle.load(adam_path)

    model = MyModel(depth, num_classes)
    optim = Adam(learning_rate=0.0001,
                 parameters=model.parameters())
    model.set_state_dict(model_state_dict)
    optim.set_state_dict(adam_state_dict)

    criterion = paddle.nn.loss.CrossEntropyLoss()

    path1 = r'data/Chromophores_features.csv'
    path2 = r'data/Solvent_features.csv'

    dataset = CustomDataset(path1, path2)
    train_atom_features = dataset.train_atom_features
    val_atom_features = dataset.val_atom_features
    test_atom_features = dataset.test_atom_features

    train_edges_nodes = dataset.train_edges_nodes
    val_edges_nodes = dataset.val_edges_nodes
    test_edges_nodes = dataset.test_edges_nodes

    train_edges_features = dataset.train_edges_features
    val_edges_features = dataset.val_edges_features
    test_edges_features = dataset.test_edges_features

    train_plqy = dataset.train_plqy
    val_plqy = dataset.val_plqy
    test_plqy = dataset.test_plqy

    train_type = dataset.train_type
    val_type = dataset.val_type
    test_type = dataset.test_type

    train_atom_features2 = dataset.train_atom_features2
    val_atom_features2 = dataset.val_atom_features2
    test_atom_features2 = dataset.test_atom_features2

    train_edges_nodes2 = dataset.train_edges_nodes2
    val_edges_nodes2 = dataset.val_edges_nodes2
    test_edges_nodes2 = dataset.test_edges_nodes2

    train_edges_features2 = dataset.train_edges_features2
    val_edges_features2 = dataset.val_edges_features2
    test_edges_features2 = dataset.test_edges_features2

    train_plqy2 = dataset.train_plqy2
    val_plqy2 = dataset.val_plqy2
    test_plqy2 = dataset.test_plqy2

    train_type2 = dataset.train_type2
    val_type2 = dataset.val_type2
    test_type2 = dataset.test_type2

    train_graphs1 = build_graphs(train_atom_features, train_edges_nodes, train_edges_features, train_type)
    train_graphs2 = build_graphs(train_atom_features2, train_edges_nodes2, train_edges_features2, train_type2)

    val_graphs1 = build_graphs(val_atom_features, val_edges_nodes, val_edges_features, val_type)
    val_graphs2 = build_graphs(val_atom_features2, val_edges_nodes2, val_edges_features2, val_type2)

    test_graphs1 = build_graphs(test_atom_features, test_edges_nodes, test_edges_features, test_type)
    test_graphs2 = build_graphs(test_atom_features2, test_edges_nodes2, test_edges_features2, test_type2)
    pre_loss = 10000
    for epoch in tqdm.tqdm(range(100)):
        start = time.time()
        train_loss, train_acc = train(train_graphs1, train_graphs2, train_type, model, len(train_graphs1), criterion, optim)
        end = time.time()
        print("epoch %d cost %s" % (epoch, str(end - start)))
        if pre_loss < train_loss:
            optim.set_lr(optim.get_lr()/10)
            print('lr reduce to %f' % (optim.get_lr()))
        else:
            pre_loss = train_loss
        print("epoch %d: train_loss %4f, train_accuracy %4f" % (epoch, train_loss, train_acc))

        val_loss, val_acc = eval(val_graphs1, val_graphs2, val_type, model, criterion)
        print("epoch %d: val_loss %4f, val_accuracy %4f" % (epoch, val_loss, val_acc))

        test_loss, test_acc = eval(test_graphs1, test_graphs2, test_type, model, criterion)
        print("epoch %d: test_loss %4f, test_accuracy %4f" % (epoch, test_loss, test_acc))

    # save
    paddle.save(model.state_dict(), "save/model.pdparams")
    paddle.save(optim.state_dict(), "save/adam.pdopt")


if __name__ == '__main__':
    model_path = r'save/model.pdparams'
    adam_path = r'save/adam.pdopt'
    load_train(model_path, adam_path)