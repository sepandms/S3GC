######################################################
####### Please define Parameters ########################
dataset_path = '../dataset/'
# dataset_name = 'Pubmed' #{'Cora', 'Citeseer', 'Pubmed','ogbn-arxiv' , 'reddit','ogbn-products'}
save_plots = False
device = 'cpu' # {'cuda','cpu'}
num_workers = None #{None, 1, 2, 3, 4,...}


cora_params     = {'walk_length':5 , 'hidden_dims':256, 'walks_per_node':20 , 'batch_size':512 , 'lr':0.01,'epochs':40,'log_steps':100}
citeseer_params = {'walk_length':5 , 'hidden_dims':256, 'walks_per_node':20 , 'batch_size':256 , 'lr':0.001,'epochs':40,'log_steps':100}
pubmed_params   = {'walk_length':5 , 'hidden_dims':256, 'walks_per_node':15 , 'batch_size':256 , 'lr':0.001,'epochs':40,'log_steps':100}
arxiv_params    = {'walk_length':5 , 'hidden_dims':256, 'walks_per_node':20 , 'batch_size':512 , 'lr':0.01,'epochs':40,'log_steps':100}
reddit_params   = {'walk_length':5 , 'hidden_dims':256, 'walks_per_node':20 , 'batch_size':256 , 'lr':0.01,'epochs':40,'log_steps':100}
products_params = {'walk_length':5 , 'hidden_dims':256, 'walks_per_node':20 , 'batch_size':512 , 'lr':0.01,'epochs':40,'log_steps':100}



####### Setting up Parameters ########################
import sys
import os
dataset_name = sys.argv[1]

model_params = {}
if str.lower(dataset_name) == 'cora':
    model_params = cora_params
elif str.lower(dataset_name) == 'citeseer':
    model_params = citeseer_params
elif str.lower(dataset_name) == 'pubmed':
    model_params = pubmed_params
elif str.lower(dataset_name) == 'ogbn-arxiv':
    model_params = arxiv_params
elif str.lower(dataset_name) == 'reddit':
    model_params = reddit_params
elif str.lower(dataset_name) == 'ogbn-products':
    model_params = products_params
else:
    print('Please define the right dataset included in the list')
    exit()

walk_length = model_params['walk_length']
hidden_dims = model_params['hidden_dims']
walks_per_node = model_params['walks_per_node']
batch_size = model_params['batch_size']
lr = model_params['lr']
epochs = model_params['epochs']
log_steps = model_params['log_steps']



embedding_store = f'embedding_{dataset_name}.pt'


if os.path.exists(dataset_path):
    None
else:
    os.mkdir(dataset_path)


if os.path.exists('../plots/'):
    None
else:
    os.mkdir('../plots/')

####### End of Setting Parameters ####################
######################################################

import argparse
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn import metrics
from torch.nn.functional import normalize
from clustering_metric import clustering_metrics
from small_model import *
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
from load_data_graph_saint import load_data



if device == 'cuda':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
print(f'############# Run S3GC for {dataset_name} #############')

def save_embedding(embedding, save_embedding):
    torch.save(embedding.cpu(), save_embedding)


def ppr(i, alpha=0.2):
    return alpha*((1-alpha)**i)
def heat(i, t=5):
    return (math.e**(-t))*(t**i)/math.factorial(i)

def norm_adj(graph):
    D = torch.sparse.mm(graph, torch.ones(graph.size(0),1).to(device)).view(-1)
    a = [[i for i in range(graph.size(0))],[i for i in range(graph.size(0))]]
    D = torch.sparse_coo_tensor(torch.tensor(a).to(device), 1/(D**0.5) , graph.size()).to(device) # D^ = Sigma A^_ii
    ADinv = torch.sparse.mm(D, torch.sparse.mm(graph, D)) # A~ = D^(-1/2) x A^ x D^(-1/2)
    return ADinv

def compute_diffusion_matrix(graph, x, niter=5, method="ppr"):
    print("Calculating S matrix")
    for i in range(0, niter):
        print("Iteration: " + str(i))
        if method=="ppr":
            theta = ppr(i)
        elif method=="heat":
            theta=heat(i)
        else:
            raise NotImplementedError
        if i==0:
            final = theta*x
            current = x
        else:
            current = torch.sparse.mm(graph, current)
            final+= (theta*current)
    return final

if str.lower(dataset_name) in ['cora','citeseer','pubmed']:

    dataset = Planetoid(root=dataset_path, name=dataset_name)
    data = dataset[0].to(device)
    del dataset
    y = data.y.view(-1)
    val_idx = data.val_mask.nonzero().view(-1).tolist()
    data.edge_index = to_undirected(add_remaining_self_loops(data.edge_index)[0])
    edge_index = data.edge_index
    normalized_A = norm_adj(torch.sparse_coo_tensor(data.edge_index.to(device), torch.ones(data.edge_index.size(1)).to(device), (data.x.size(0),data.x.size(0))))
    AX = torch.sparse.mm(normalized_A, data.x)
    SX = compute_diffusion_matrix(normalized_A, data.x, niter=3)
    del normalized_A, data

elif str.lower(dataset_name) in ['ogbn-arxiv','ogbn-products']:

    dataset = PygNodePropPredDataset(name = dataset_name, root=dataset_path)
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    data = dataset[0].to(device)
    del dataset
    y = data.y.view(-1).cpu()
    data.edge_index = to_undirected(add_remaining_self_loops(data.edge_index)[0])
    edge_index = data.edge_index
    normalized_A = norm_adj(torch.sparse_coo_tensor(data.edge_index.to(device), torch.ones(data.edge_index.size(1)).to(device), (data.x.size(0),data.x.size(0))))
    AX = torch.sparse.mm(normalized_A, data.x)
    SX = compute_diffusion_matrix(normalized_A, data.x, niter=3)
    del normalized_A, data

elif str.lower(dataset_name) in ['reddit']:
    A,_,X,label,split = load_data(dataset_path + dataset_name)
    val_idx = split['va']
    y = label.view(-1).cpu()
    A = A.to(device)
    X = X.float().to(device)
    edge_index = A.coalesce().indices()
    edge_index = to_undirected(add_remaining_self_loops(edge_index)[0])
    normalized_A = norm_adj(torch.sparse_coo_tensor(edge_index.to(device), torch.ones(edge_index.size(1)).to(device), (X.size(0),X.size(0))))
    AX = torch.sparse.mm(normalized_A, X)
    SX = compute_diffusion_matrix(normalized_A, X, niter=3)
    del normalized_A, X, A, split


model = Node2Vec(AX.size(-1), hidden_dims, AX.size(0), 0.0, edge_index, hidden_dims, walk_length,
                    walk_length, walks_per_node, p=1.0, q=1.0,
                    sparse=True).to(device)

if num_workers:
    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
else:
    loader = model.loader(batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def main():

    model.train()
    start_train=time.time()
    mapping = torch.zeros(AX.size(0), dtype=torch.int64).to(device)
    kmeans = KMeans(n_clusters=y.max().item()+1, n_init=20)
    best_epoch = -1
    max_nmi = -1
    nmi_list = []
    details_val = pd.DataFrame(columns=['Epoch','Accuracy','NMI','CS','F1','ARI','loss']) 
    details_train = pd.DataFrame(columns=['Epoch','Accuracy','NMI','CS','F1','ARI','loss']) 
    for epoch in range(1, epochs + 1):
    # for epoch in range(start_epoch, 3 + 1):
        start_epoch= time.time()
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            pos_rw = pos_rw.to(device)
            neg_rw = neg_rw.to(device)
            unique = torch.unique(torch.cat((pos_rw, neg_rw), dim=-1))
            mapping.scatter_(0, unique, torch.arange(unique.size(0)).to(device))
            model.update_B(F.embedding(unique, AX), F.embedding(unique, SX), unique)
            loss = model.loss(pos_rw, neg_rw, mapping)
            loss.backward()
            optimizer.step()

            if (i + 1) % log_steps == 0:
                print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, 'f'Loss: {loss:.4f}')

        # validation dataset
        embedding = model.get_embedding(AX, SX)
        y_pred = kmeans.fit_predict(embedding.detach()[val_idx].cpu().numpy())
        cm = clustering_metrics(y[val_idx].cpu().numpy(), y_pred)
        acc, nmi, cs, f1_macro, adjscore  = cm.get_main_metrics()

        new_row = {'Epoch':epoch,'Accuracy':acc,'NMI':nmi,'CS':cs,'F1':f1_macro,'ARI':adjscore}
        details_val = details_val.append(new_row,ignore_index=True)

        print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, 'f'Loss: {loss:.4f}')
        print("Time taken for this epoch: " + str(time.time()-start_epoch))
        print("Acc: ", acc, "NMI: ", nmi, "cs: ", cs, "f1: ", f1_macro, "adjscore: ", adjscore)

        # y_pred_ = kmeans.fit_predict(embedding.detach().cpu().numpy())
        # cm = clustering_metrics(y.cpu().numpy(), y_pred_)

        y_pred = kmeans.fit_predict(embedding.detach().cpu().numpy())
        cm = clustering_metrics(y.cpu().numpy(), y_pred)
        acc, nmi, cs, f1_macro, adjscore  = cm.get_main_metrics()

        new_row = {'Epoch':epoch,'Accuracy':acc,'NMI':nmi,'CS':cs,'F1':f1_macro,'ARI':adjscore}
        details_train = details_train.append(new_row,ignore_index=True)


        if nmi > max_nmi:
            best_epoch = epoch
            max_nmi = nmi
            save_embedding(embedding, embedding_store)
        del embedding, y_pred

    print("Time taken for training: " + str(time.time()-start_train))
    print("######################### END of Training ####################")
    embedding = torch.load(embedding_store)
    y_pred = kmeans.fit_predict(embedding.detach().cpu().numpy())
    cm = clustering_metrics(y.cpu().numpy(), y_pred)
    acc, nmi, cs, f1_macro, adjscore  = cm.get_main_metrics()
    print(' ############### Performance Results on  whole dataset ###############')
    print("Acc: ", acc, "NMI: ", nmi, "cs: ", cs, "f1: ", f1_macro, "adjscore: ", adjscore)

    y_pred = kmeans.fit_predict(embedding.detach()[val_idx].cpu().numpy())
    cm = clustering_metrics(y[val_idx].cpu().numpy(), y_pred)
    acc, nmi, cs, f1_macro, adjscore  = cm.get_main_metrics()
    print(' ############### Performance Results on  Valid dataset ###############')
    print("Acc: ", acc, "NMI: ", nmi, "cs: ", cs, "f1: ", f1_macro, "adjscore: ", adjscore)

    os.remove(embedding_store)

    # ploting and saving the plots

    fig, ax = plt.subplots(figsize=(6,4))
    KPIs = ['Accuracy','NMI','CS','F1','ARI']
    plot_data = details_val[KPIs]
    epochs_X = details_val['Epoch']
    # plt.subplots(figsize=(6,4))
    for m_ in KPIs:
            plt.plot(epochs_X, plot_data[m_ ], 'o-', label=m_)
    plt.xlabel("Epochs", fontsize = 12)
    plt.ylabel("Metrichs", fontsize = 12)
    plt.grid()
    plt.legend(loc='lower center',ncol=5)
    plt.title(f'S3GC: Performance Metrichs ({dataset_name}: Valid)')
    plot_name = f'S3GC_{dataset_name}_Valid.png'
    if save_plots:
        plt.savefig('../plots/'+plot_name)
    # plt.show()

    fig, ax = plt.subplots(figsize=(6,4))
    KPIs = ['Accuracy','NMI','CS','F1','ARI']
    plot_data = details_train[KPIs]
    epochs_X = details_train['Epoch']
    # plt.subplots(figsize=(6,4))
    for m_ in KPIs:
            plt.plot(epochs_X, plot_data[m_ ], 'o-', label=m_)
    plt.xlabel("Epochs", fontsize = 12)
    plt.ylabel("Metrichs", fontsize = 12)
    plt.grid()
    plt.legend(loc='lower center',ncol=5)
    plt.title(f'S3GC: Performance Metrichs ({dataset_name}: Train)')
    plot_name = f'S3GC_{dataset_name}_Train.png'
    if save_plots:
        plt.savefig('../plots/'+plot_name)
    # plt.show()


if __name__ == "__main__":
    main()
