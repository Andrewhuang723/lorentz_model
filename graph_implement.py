import dgl
import torch
import pickle
import networkx as nx
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda")

T = 10
data_size = 100000
t = np.linspace(0, T, data_size)
b = 8 / 3
sig = 10
r = 28

def Lorenz(state, t):
    x, y, z = state  # Unpack the state vector
    return sig * (y - x), x * (r - z) - y, x * y - b * z  # Derivatives

data = []
for i in range(100):
    np.random.seed(i) #random with sorting
    x0 = 30 * (np.random.rand(3) - 0.5)
    states = odeint(Lorenz, x0, t)
    data.append(states)
data = np.asarray(data)
data = torch.Tensor(data) #(100, 100000, 3)

data_1 = data[2] #(100000, 3)
data_2 = data[7]
test_data = data #(100, 100000, 3)

#torch.save(test_data, "./lorentz_data/test_data.pkl")

from Model import ode2graph

nodes = 3
out_message_pair = [[0,1], [1,2], [0,2]]
in_message_pair = [[0,0], [1,1], [2,2]]

g, etype = ode2graph(nodes=nodes, states=data_1[0], in_message_pair=in_message_pair, out_message_pair=out_message_pair)
print(g, g.ndata["h"], g.edata["h"], etype)

nx.draw(g.to_networkx(), with_labels=True)
plt.show()

'''
train_graphs = []
for i in range(len(data_1)):
    try:
        graph, _ = ode2graph(nodes=nodes, states=data_1[i], in_message_pair=in_message_pair, out_message_pair=out_message_pair)
        train_graphs.append(graph)
    except:
        pass
dgl.data.save_graphs("./lorentz_data/train_graphs.bin", train_graphs)

test_graphs_1 = []
for i in range(len(test_data[7])):
    try:
        test_graph, _ = ode2graph(nodes=nodes, states=test_data[7][i],  in_message_pair=in_message_pair, out_message_pair=out_message_pair)
        test_graphs_1.append(test_graph)
    except:
        pass
dgl.data.save_graphs("./lorentz_data/graphs_7.bin", test_graphs_1)
'''

train_graphs_1, _ = dgl.data.load_graphs("lorentz_data/graphs_2.bin")
train_graphs_2, _ = dgl.data.load_graphs("lorentz_data/graphs_7.bin")
print(train_graphs_1[1].ndata["h"], data_1[1])
print(train_graphs_2[1].ndata["h"], data_2[1])

def residual(data):
    res = []
    for i in range(len(data)):
        if i == len(data) - 1:
            break
        res.append(data[i+1] - data[i])
    return torch.stack(res)

index = list(range(0, len(train_graphs_1), 10))
X = train_graphs_1[:-1] + train_graphs_2[:-1]
y = torch.cat([residual(data_1)[index], residual(data_2)[index]], dim=0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=True)
print("X_train: ", len(X_train), "Y_train: ", y_train.shape, "\nX_val: ", len(X_val), "y_val: ", y_val.shape)

from torch.utils.data import DataLoader
from Model import time_dataset, collate

train_loader = DataLoader(dataset=time_dataset(X_train, y_train), batch_size=512, shuffle=False, collate_fn=collate)
val_loader = DataLoader(dataset=time_dataset(X_val, y_val), batch_size=512, shuffle=False, collate_fn=collate)

from Model import GGNN
from time import time
from torch import optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.nn import MSELoss
from pytorch_tools import EarlyStopping

ode_gnn = GGNN(in_feats=4, n_hidden=10, out_feats=20, n_iter_readout=3).cuda()
optimizer = optim.Adam(ode_gnn.parameters(), lr=0.01)
sched = StepLR(optimizer=optimizer, step_size=30, gamma=0.1)
sched = MultiStepLR(optimizer=optimizer, milestones=[10, 40], gamma=0.1)
criterion = MSELoss().cuda()
early_stopping = EarlyStopping(patience=30)

def train(model, epochs, loss_func, data_loader, val_data_loader):
    train_epochs_losses = []
    val_epoch_losses = []
    dur = []
    for epoch in range(epochs):
        model.train()
        train_epochs_loss = 0
        if epoch >= 1:
            t0 = time()
        for graph, state in data_loader:
            graph = graph.to(device)
            state = torch.stack(state).to(device, dtype=torch.float)
            prediction = model(graph, graph.ndata["h"], graph.edata["h"]) #(batch_size * 3, 1)
            loss = 0
            for i in range(len(prediction)):
                loss += loss_func(prediction[i, :], state[i, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epochs_loss += (loss / len(prediction))
        train_epochs_loss /= len(data_loader)

        model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for val_g, val_state in val_data_loader:
                val_g = val_g.to(device)
                val_state = torch.stack(val_state).to(device, dtype=torch.float)
                val_pre = model(val_g, val_g.ndata["h"], val_g.edata["h"])
                val_loss = 0
                for i in range(len(val_pre)):
                    val_loss += loss_func(val_pre[i, :], val_state[i, :])
                val_epoch_loss += (val_loss / len(val_pre))
            val_epoch_loss /= len(val_data_loader)

        sched.step()

        if epoch >= 1:
            dur.append(time() - t0)

        early_stopping(val_loss=val_epoch_loss, model=model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print('Epoch {} | loss {:.10f} | Time(s) {:.4f} | val_loss {:.10f}'.format(epoch, train_epochs_loss,
                                                                                 np.mean(dur),
                                                                                 val_epoch_loss))
        train_epochs_losses.append(train_epochs_loss)
        val_epoch_losses.append(val_epoch_loss)
    nn_dict = model.state_dict()
    nn_dict["loss"] = train_epochs_losses
    nn_dict["val_loss"] = val_epoch_losses
    return nn_dict

model_dict = train(model=ode_gnn, epochs=1000, loss_func=criterion, data_loader=train_loader, val_data_loader=val_loader)
torch.save(model_dict, "./lorentz_converge_set2set_2.pkl")