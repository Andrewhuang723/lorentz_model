import dgl
from dgl.nn.pytorch import GatedGraphConv, Set2Set, GraphConv
from dgl.nn.pytorch import NNConv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

device = torch.device("cuda")

def edge_type(edge, labels):
    return [edge == i for i in labels]

def ode2graph(nodes, states, in_message_pair, out_message_pair):
    g = dgl.DGLGraph()
    g.add_nodes(nodes)
    A = torch.Tensor([[0, 1, 0],
                     [1, 0, 1],
                     [1, 1, 0]])
    n_feat = torch.cat([states.view(-1, 1), A], dim=1)
    income = []; outcome = []
    all_edges = in_message_pair + out_message_pair
    e_feat = []
    for i, j in zip(in_message_pair, out_message_pair):
        income.extend(i)
        outcome.extend(j)
        e_feat.extend(torch.Tensor([edge_type(i, labels=all_edges), edge_type(j, labels=all_edges)]))

    g.add_edges(income, outcome)

    g.edata["h"] = torch.Tensor([a.tolist() for a in e_feat])
    g.ndata["h"] = n_feat
    etype = torch.tensor(range(len(all_edges))).int()
    return g, etype

class time_dataset(Dataset):
    def __init__(self, graphs, states):
        self.graphs = graphs
        self.states = states

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        return self.graphs[item], self.states[item]

def collate(samples):
    graphs, states = map(list, zip(*samples))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return bg, states

class Set2Set(nn.Module):

    def __init__(self, input_dim, n_iters, n_layers):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = torch.nn.LSTM(input_size=self.output_dim, hidden_size=self.input_dim, num_layers=n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        self.lstm.reset_parameters()

    def forward(self, graph, feat):
        with graph.local_scope():
            batch_size = graph.batch_size

            h = (feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
                 feat.new_zeros((self.n_layers, batch_size, self.input_dim))) #(6, 32, 100)

            q_star = feat.new_zeros(batch_size, self.output_dim) #(32, 200)
            #print(q_star.shape)
            for i in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feat * dgl.broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True)

                graph.ndata['e'] = e
                alpha = dgl.softmax_nodes(graph, 'e')
                graph.ndata['r'] = feat * alpha
                readout = dgl.sum_nodes(graph, 'r')
                q_star = torch.cat([q, readout], dim=-1)

            return q_star
class GGNN(nn.Module):
    def __init__(self, in_feats, out_feats, n_hidden, n_iter_readout):
        super(GGNN, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden = n_hidden
        self.conv1 = GatedGraphConv(in_feats=in_feats, out_feats=out_feats, n_etypes=6, n_steps=5).cuda()
        self.conv2 = GraphConv(in_feats=in_feats, out_feats=out_feats)
        self.edge_net = nn.Linear(6, n_hidden)
        self.feat_net = nn.Linear(in_feats, n_hidden)
        self.lin_1 = nn.Linear(out_feats, n_hidden)
        self.lstm = nn.LSTM(input_size=n_hidden, hidden_size=n_hidden, num_layers=3, batch_first=True)
        self.lin_2 = nn.Linear(in_features=n_hidden, out_features=3)
        self.predict = nn.Linear(2 * out_feats, 3)
        self.dropout = nn.Dropout(p=0.1)
        self.set2set = Set2Set(input_dim=out_feats, n_iters=n_iter_readout, n_layers=3)

    '''
    def forward(self, g, feats, edge_feats):
        prop_conv = [self.conv21, self.conv31, self.conv41, self.conv51]
        out = self.conv1(g, feats, edge_feats)
        for i in range(len(prop_conv)):
            out = prop_conv[i](g, out, edge_feats) #(384, 10)
            out = out.view(g.batch_size, 3, -1)
            out = out.transpose(3, g.batch_size, -1)
            g, _ = ode2graph(nodes=nodes, states=out, in_message_pair=in_message_pair, out_message_pair=out_message_pair)
        out = self.lin_1(out)
        out = self.predict(out)
        return out

    def forward(self, g, feats, edge_feats):
        with g.local_scope():
            batch_size = g.batch_size
            out_feat = self.conv1(g, feats, edge_feats) #(3 * batch_size, out_feats) --> H(1)
            h = out_feat #(batch_size * 3, out_feats)
            #x_i = dgl.broadcast_nodes(g, h.view(g.batch_size, -1)).sum(dim=-1, keepdim=True)  #(H(1) --> X(2): shape = (3 * batch_size, 1))
            #x_i = nn.Linear(self.out_feats, 1).cuda()(h) #(H(1) --> X(2))
            out_conv = [self.conv2, self.conv3, self.conv4, self.conv5]
            propagate_conv = [self.conv21, self.conv31, self.conv41, self.conv51]
            for i in range(self.output_dim - 1):
                out_feat = torch.cat([out_feat, out_conv[i](g, h, edge_feats)], dim=0) #(5 * batch_size * 3, out_feats) --> H(2)~H(5)
                h = propagate_conv[i](g, h, edge_feats)
                #x_i = dgl.broadcast_nodes(g, h.view(g.batch_size, -1)).sum(dim=-1, keepdim=True)  #(H(i) --> X(i+1): shape = (3 * batch_size, 1))
                #x_i = nn.Linear(self.out_feats, 1).cuda()(h)
            out_feat = out_feat.reshape(self.output_dim, batch_size, -1, self.out_feats)
            out = out_feat.transpose(1, 0)
            out = self.lin_1(out)
            out = self.dropout(out)
            out = self.predict(out)
            out = out.squeeze(3)
        return out
'''

    def forward(self, g, feats, edge_feats):
        batch_size = g.batch_size
        #feats = self.feat_net(feats)
        out = self.conv1(g, feats, edge_feats) #(batch_size * 3, out_feats)
        out = self.set2set(g, out)
        #out = out.view(batch_size, 3, -1)
        out = self.predict(out)

        return out

