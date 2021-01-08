# testing data is the rest randomize initial states
import pickle
import torch
import numpy as np
import torch.nn as nn
import dgl
import os
import matplotlib.pyplot as plt
from Model import GGNN
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda")
dict = torch.load("./lorentz_converge_set2set_2.pkl")

loss = dict["loss"]
print(min(loss))
val_loss = dict["val_loss"]
plt.figure()
plt.plot(range(len(loss)), loss, label="Training")
plt.plot(range(len(val_loss)), val_loss, label="Validation")
plt.legend(loc="upper right")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.title("lorentz_gnn")
plt.show()

test_graphs, _ = dgl.data.load_graphs("lorentz_data/graphs_2.bin")
test_data = torch.load("./lorentz_data/test_data.pkl")
y_test_1 = test_data[2].cuda()

from torch.utils.data import DataLoader
from Model import time_dataset, collate, GGNN
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

model = GGNN(in_feats=4, n_hidden=10, out_feats=20, n_iter_readout=3).cuda()
del dict["loss"]
del dict["val_loss"]
model.load_state_dict(dict)
loss_func = nn.MSELoss().cuda()

nodes = 3
out_message_pair = [[0,1], [1,2], [0,2]]
in_message_pair = [[1,0], [2,1], [2,0]]

def edge_type(edge, labels):
    return [edge == i for i in labels]

def ode2graph(nodes, states, in_message_pair, out_message_pair):
    g = dgl.DGLGraph().to(device)
    g.add_nodes(nodes)
    A = torch.Tensor([[0, 1, 0],
                     [1, 0, 1],
                     [1, 1, 0]]) # no node annotations yet
    n_feat = torch.cat([states.view(-1, 1), A.cuda()], dim=1)
    income = []; outcome = []
    all_edges = in_message_pair + out_message_pair
    e_feat = []
    for i, j in zip(in_message_pair, out_message_pair):
        income.extend(i)
        outcome.extend(j)
        e_feat.extend(torch.Tensor([edge_type(i, labels=all_edges), edge_type(j, labels=all_edges)]))

    g.add_edges(income, outcome)

    g.edata["h"] = torch.Tensor([a.tolist() for a in e_feat]).cuda()
    g.ndata["h"] = n_feat
    etype = torch.tensor(range(len(all_edges))).int()
    return g, etype

def plot(y_pred, y_test, R_square, MAE, data_columns):
    plt.subplots(figsize=(9, 7))
    fig = plt.figure()
    for i in range(3):
        plt.scatter(y_pred[:, i], y_test[:, i], c='g', s=2, label="Prediction")
        plt.plot(y_test[:, i], y_test[:, i], color='k', ls="--", linewidth=1, label="Testing")
        plt.legend(loc="upper right")
        plt.title(data_columns + "\nR_square: {:.4f} \nmae: {:.4f} ".format(R_square, MAE))
        plt.xlabel('Prediction')
        plt.ylabel('Testing sets')
        plt.show()

def evaluate_1(X_test, y_test, loss_func):
    g = X_test[0].to(device)
    loss = 0
    pred_df = g.ndata["h"][:, 0].view(1, -1)  # (1, 3)
    iter = len(X_test) - 1
    for i in range(iter):
        y_pred = model(g, g.ndata["h"], g.edata["h"])
        y_pred += g.ndata["h"][:, 0].view(1, -1)
        g, _ = ode2graph(nodes=3, states=y_pred, in_message_pair=in_message_pair, out_message_pair=out_message_pair)
        loss += loss_func(y_pred, y_test[i + 1])
        pred_df = torch.cat([pred_df, y_pred], dim=0)

        if i % 1000 == 0 and i > 0:
            print(loss / (len(pred_df) - 1))
            MAE = mean_absolute_error(pred_df[1: i + 2].cpu().detach().numpy(), y_test[1: i + 2].cpu().detach().numpy())
            RSQUARE = r2_score(pred_df[1: i + 2].cpu().detach().numpy(), y_test[1: i + 2].cpu().detach().numpy())
            plot(pred_df[1: i + 2].cpu().detach().numpy(), y_test[1: i + 2].cpu().detach().numpy(), R_square=RSQUARE, MAE=MAE, data_columns="TEST: " + str(i))
    loss /= (len(pred_df) - 1)
    return loss, pred_df

loss, prediction = evaluate_1(X_test=test_graphs, y_test=y_test_1, loss_func=loss_func)
print(prediction.shape)
prediction_df = pd.DataFrame(prediction)
prediction_df.to_excel("./lorentz_data/GGNN_pred.xlsx")


MAE = mean_absolute_error(prediction, y_test_1[1:])
RSQUARE = r2_score(prediction, y_test_1[1:])



plot(prediction, y_test_1, RSQUARE, MAE, "TEST: 23")


import pandas as pd
prediction_df = pd.DataFrame(prediction)
#prediction_df.to_excel("./prediction.xlsx")



from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.scatter(y_test_1[:, 0], y_test_1[:, 1], y_test_1[:, 2], s=1, c="r", marker="o", label="y_test")
ax.scatter(prediction[:, 0], prediction[:, 1], prediction[:, 2], s=1, c="g", label="y_pred")
ax.legend()
plt.title("Testing Loss:" + "%.6f" % loss)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
