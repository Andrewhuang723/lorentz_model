# Lorentz model
Using GNN to simulate Lorentz ODE

**Requirements: Pytorch 1.2.0, DGL 0.5.3**

## graph_implement.py

Design graph from [DGL 0.5.3](https://docs.dgl.ai/), [Pytorch 1.2.0](https://pytorch.org/docs/stable/index.html)

### Lorenz(state, t): 

_FUNCTION_

_output: (x, y, z) which is the Lorentz ODE_

### residual(data): 

_FUNCTION_

_output: torch.Tensor_

Calculating the difference between each time step, dx/dt...


### train(model, epochs, loss_func, data_loader, val_data_loader): 

_FUNCTION_

_output: dict (model parameters)_

Model training...

## Model.py

Construct Graph neyral network from [DGL 0.5.3](https://docs.dgl.ai/), [Pytorch 1.2.0](https://pytorch.org/docs/stable/index.html)

A set of classes and functions would be implemented

### edge_type(edge, labels): 

_FUNCTION_

_output: torch.Tensor_

Labeled the edges into one-hot vectors

### ode2graph(nodes, states, in_message_pair, out_message_pair):

_FUNCTION_

_output: dgl.graph_

Construct and implement graph features into dgl.graph

## GGNN(in_feats, out_feats, n_hidden, n_iter_readout): 

_CLASS_

```
def forward(self, g, feats, edge_feats):
    batch_size = g.batch_size
    out = self.conv1(g, feats, edge_feats) 
    out = self.set2set(g, out)
    out = self.predict(out)
```
_output: torch.Tensor_

Construct the model based on [gated graph neural network](https://arxiv.org/pdf/1511.05493.pdf) and readout function [set2set](https://arxiv.org/pdf/1511.06391.pdf),
this is similar to the [message passing neural network](https://arxiv.org/pdf/1704.01212.pdf) proposed by [J.Gilmer et.al](https://www.linkedin.com/in/jmgilmer/)

## evaluate.py

In this file, the prediction is based on one-to-one prediction, and evaluate by mean-squared error. An output is an input of the model at the next time step 
