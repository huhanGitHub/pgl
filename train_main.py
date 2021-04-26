import pandas as pd
from my_dataset import CustomDataset
from custom_gcn import MyModel
import paddle
import tqdm
import time
from paddle.optimizer import Adam
import pgl
import numpy as np
import math
from visualdl import LogWriter
from visualdl.server import app


@paddle.no_grad()
def eval(eval_pairs, model, criterion):
    model.eval()
    preds = []
    labels = []
    for pair in eval_pairs:
        g1 = pair[0]
        g2 = pair[1]
        labels.append(pair[2])
        pred = model(g1, g1.node_feat['nfeat'], g2, g2.node_feat['nfeat'])
        pred = paddle.unsqueeze(pred, axis=0)
        preds.append(pred)

    preds = paddle.concat(preds, axis=0)
    labels = paddle.to_tensor(labels)
    labels = paddle.unsqueeze(labels, axis=1)
    loss = criterion(preds, labels)
    acc = math.log(loss.numpy().mean()/len(labels))
    # acc = paddle.static.accuracy(input=preds, label=labels)
    return loss.numpy()[0], acc


def train(train_pairs, model, batch_size, criterion, optim):
    model.train()
    # batch_size = 64
    batch_train_pairs = split_by_size(train_pairs, batch_size)
    step = 0
    loss_epoch = []
    acc_epoch = []
    for batch_train_pair in batch_train_pairs:
        preds = []
        batch_label = []
        for pair in batch_train_pair:
            g1 = pair[0]
            g2 = pair[1]
            batch_label.append(pair[2])
            pred = model(g1, g1.node_feat['nfeat'], g2, g2.node_feat['nfeat'])
            pred = paddle.unsqueeze(pred, axis=0)
            preds.append(pred)

        preds = paddle.concat(preds, axis=0)
        batch_label = paddle.to_tensor(batch_label, dtype='float32')
        loss = criterion(preds, batch_label)
        # avg_loss = paddle.mean(loss)
        loss_epoch.append(loss.numpy())
        loss.backward()
        optim.step()
        batch_label = paddle.unsqueeze(batch_label, axis=1)
        # acc = paddle.static.accuracy(input=preds, label=batch_label)
        acc = math.log(loss.numpy().mean()/len(batch_label))
        acc_epoch.append(acc)
        optim.clear_grad()
        # print("step: %s | train_loss: %.4f | train_accuracy: %.4f" % (step, loss.numpy()[0], acc))
        step += 1

    return sum(loss_epoch)/len(loss_epoch), min(acc_epoch)


def main():
    batch_size = 256
    depth = 44
    epoch = 1000
    dur = []
    best_test = []
    path1 = r'data/Chromophores_features.csv'
    path2 = r'data/Solvent_features.csv'

    dataset = CustomDataset(path1, path2)
    num_classes = dataset.num_class
    model = MyModel(depth, num_classes)
    # criterion = paddle.nn.loss.CrossEntropyLoss()
    criterion = paddle.nn.MSELoss(reduction='mean')
    optim = Adam(learning_rate=0.01,
                 parameters=model.parameters(), weight_decay=paddle.regularizer.L2Decay(0.0001))
    pre_loss = 10000
    with LogWriter(logdir="./log/scalar_test/train") as writer:
        for epoch in tqdm.tqdm(range(500)):
            train_pairs, val_pairs, test_pairs = dataset.get_train_val_test_pairs()
            start = time.time()
            train_loss, train_acc = train(train_pairs, model, batch_size, criterion, optim)
            end = time.time()
            print("epoch %d cost %s" % (epoch, str(end - start)))
            if pre_loss < train_loss and optim.get_lr() >= 0.00001:
                optim.set_lr(optim.get_lr()/2)
                print('lr reduce to %f' % (optim.get_lr()))
            else:
                pre_loss = train_loss
            writer.add_scalar(tag="train_loss", step=epoch, value=train_loss)
            print("epoch %d: train_loss %4f, train_accuracy %4f" % (epoch, train_loss, train_acc))

            val_loss, val_acc = eval(val_pairs, model, criterion)
            print("epoch %d: val_loss %4f, val_accuracy %4f" % (epoch, val_loss, val_acc))
            writer.add_scalar(tag="val_loss", step=epoch, value=val_loss)

            test_loss, test_acc = eval(test_pairs, model, criterion)
            print("epoch %d: test_loss %4f, test_accuracy %4f" % (epoch, test_loss, test_acc))
            writer.add_scalar(tag="test_loss", step=epoch, value=test_loss)

            writer.add_scalar(tag="lr", step=epoch, value=optim.get_lr())

            if epoch >= 50 and epoch % 50 == 0:
                # save
                paddle.save(model.state_dict(), "save/model.pdparams")
                paddle.save(optim.state_dict(), "save/adam.pdopt")
                print('data of epoch %d have saved' % epoch)


def split_by_size(arr, size):
    s = []
    for i in range(0, int(len(arr)), size):
        c = arr[i:i + size]
        s.append(c)
    return s


def test():
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    print(split_by_size(a, 6))

if __name__ == '__main__':
    #data_loader(r'data/Chromophores_features.csv')
    main()