import tqdm
import sklearn.metrics
from numba import jit
import dgl.nn as dglnn
import dgl.function as fn
from sympy import I
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import numpy as np
import dgl
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import copy
import random
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from numba import jit
import warnings
# from GATLayer import WSWGAT
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing
import torch.utils.data as Data
import scipy.sparse as sp
import networkx as nx
import xgboost as xgb
import numpy as np
import pandas, scipy
from xgboost_ray import RayDMatrix, RayParams, predict, train, RayXGBClassifier
from sklearn.datasets import load_breast_cancer
import xgboost as xgb
import pp
import sys
import scipy.stats as stats
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from numba import cuda
import torch.multiprocessing as mp
 

class NegativeSampler(object):
    def __init__(self, g, k):
        # caches the probability distribution
        self.weights = {
            etype: g.in_degrees(etype=etype).float() ** 0.75
            for etype in g.canonical_etypes}
        self.k = k

    def __call__(self, g, eids_dict):
        result_dict = {}
        for etype, eids in eids_dict.items():
            src, _ = g.find_edges(eids, etype=etype)
            src = src.repeat_interleave(self.k)
            dst = self.weights[etype].multinomial(len(src), replacement=True)
            result_dict[etype] = (src, dst)
        return result_dict

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope(): 
            graph.ndata['h'] = h 
            
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']
 

class StochasticThreeLayerRGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, hidden_feat_1,out_feat, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
                rel : dglnn.GraphConv(in_feat, hidden_feat, norm='right')
                for rel in rel_names
            })
        self.conv2 = dglnn.HeteroGraphConv({
                rel : dglnn.GraphConv(hidden_feat, hidden_feat_1, norm='right')
                for rel in rel_names
            })
        self.conv3 = dglnn.HeteroGraphConv({
                rel : dglnn.GraphConv(hidden_feat_1, out_feat, norm='right')
                for rel in rel_names
            })
        #self.conv4 = dglnn.HeteroGraphConv({
        #        rel : dglnn.GraphConv(hidden_feat_1, out_feat, norm='right')
        #        for rel in rel_names
        #    })
        #self.conv5 = dglnn.HeteroGraphConv({
        #        rel : dglnn.GraphConv(hidden_feat_1, out_feat, norm='right')
        #        for rel in rel_names
        #    })

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)     
        x = {k: F.relu(v) for k, v in x.items()}     
        x1 = self.conv2(blocks[1], x)
        x = {k: F.relu(v) for k, v in x1.items()} 
        x = self.conv3(blocks[2], x) 
        x = {k: F.relu(v) for k, v in x.items()}
        return x, x1
    
def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad(): 
        logits = model.module.rgcn([graph,graph,graph], features)[0]['paper'] #module.
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['features'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('features', 'features', 'score'), etype=etype)
            return edge_subgraph.edata['score']

        
def compute_loss(pos_score, neg_score):
    # an example hinge loss
    pos_score_1=[]
    for key in pos_score:
        pos_score_1.extend(pos_score[key])
    pos_score_1=torch.stack(pos_score_1)  
    neg_score_1=[]
    for key in neg_score:
        neg_score_1.extend(neg_score[key])
    neg_score_1=torch.stack(neg_score_1) 
    n = pos_score_1.shape[0]
    return (neg_score_1.view(n, -1) - pos_score_1.view(n, -1) + 1).clamp(min=0).mean()  
  
class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            print ( x)
            edge_subgraph.ndata['features'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('features', 'features', 'score'), etype=etype)
            return edge_subgraph.edata['score']
#logits = model_lp.sage(hetero_graph.to(torch.device('cuda')), node_features)['paper']
 

 
   

def run(proc_id,device_ids,  model_lp,  hetero_graph ,  train_mask, 
       max_epoch):
    #torch.distributed.init_process_group(backend='nccl',   init_method="env://",rank=0, world_size=1)
    print ('proc_id',proc_id)
    dev_id = device_ids[proc_id] 
    device = torch.device('cuda:' + str(dev_id))
    
 
    
    hetero_graph=hetero_graph.to(device)
    #train_eid_dict = {
    #        etype: hetero_graph.edges(etype=etype, form='eid')
    #        for etype in [('paper','cocit','paper' ) ,( 'paper','link','class' )]}
    
    #sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    #sampler = dgl.dataloading.as_edge_prediction_sampler(
    #        sampler, negative_sampler=NegativeSampler(hetero_graph, 5))
    #dataloader = dgl.dataloading.DataLoader(
    #        hetero_graph, train_eid_dict, sampler,
    #        batch_size=1024*60000 ,  
    #        shuffle=True,
    #        drop_last=False,
    #        num_workers=0,
    #        use_ddp=True,
            #use_uva=True,
    #        device=device )   # *200 for MAG*60000*3
    
    train_nid_dict = {
            ntype: hetero_graph.nodes(ntype=ntype )
            for ntype in [ 'paper','ref'    ]}
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3) 
    dataloader_1 = dgl.dataloading.DataLoader(
        hetero_graph, train_nid_dict, sampler,
        batch_size=1024*60000,
        shuffle=True,
        drop_last=False,
        num_workers=0,
            use_ddp=True,
            #use_uva=True,
            device=device )
    
    #train_eid_dict_1 = {
    #        etype: hetero_graph.edges(etype=etype, form='eid')
    #        for etype in [( 'paper','link','class' ) ]}
    
    #sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    #sampler = dgl.dataloading.as_edge_prediction_sampler(
    #        sampler, negative_sampler=NegativeSampler(hetero_graph, 5))
    #dataloader_2 = dgl.dataloading.DataLoader(
    #        hetero_graph, train_eid_dict_1, sampler,
    #        batch_size=1024*6000 ,  
    #        shuffle=True,
    #        drop_last=False,
    #        num_workers=0,
    #        use_ddp=True,
            #use_uva=True,
    #        device=device )   # *200 for MAG*60000*3
    
    
    
    
  
     
    train_mask= train_mask.to(device)
    #model_lp = Model_LP(n_features,512, hidden_feat_1, num_class, hetero_graph.etypes)   
    model_lp = model_lp.to(device )
    #model_lp = DDP(model_lp, device_ids=[device],  output_device='cuda:0',find_unused_parameters=True ) #, device_ids=device_ids   
    
    model_lp = DDP(model_lp, device_ids=[device],  output_device=torch.device('cuda:' + str(device_ids[0] )),find_unused_parameters=True ) #, device_ids=device_ids     
    opt = torch.optim.Adam(model_lp.parameters())
    torch.autograd.set_detect_anomaly(True) 

    
  
    #if os.path.exists(model_path):
    #    model_lp.load_state_dict(torch.load(model_path)) 
    #else:
    loss1=[]
    for epoch in range(max_epoch):
        print (epoch) 
        model_lp.train() 
        #with tqdm.tqdm(dataloader ) as tq:
        #    task_tag=1
        #    for step, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(tq ):
                
        #        blocks = [b.to(device) for b in blocks] 
        #        positive_graph = positive_graph.to(device)
        #        negative_graph = negative_graph.to(device)
        #        input_features = blocks[0].srcdata['features'] 
        #        print (input_features['paper'].size(),input_features['class'].size()) 
        #        output_labels = blocks[-1].dstdata['label']['paper'] 
        #        loss= model_lp(positive_graph, negative_graph, blocks,  input_features, output_labels,  train_mask,device,task_tag)
                    
                    
        #        loss1.append(loss.item())
        #            #print (loss.item())
        #        opt.zero_grad()
        #        loss.backward()
        #        opt.step() 
        with tqdm.tqdm(dataloader_1 ) as tq:
            task_tag=2
            for step, (input_nodes, output_nodes, blocks) in enumerate(tq):
                blocks = [b.to(device) for b in blocks] 
                positive_graph=[]
                negative_graph=[] 
                input_features = blocks[0].srcdata['features']
                output_labels = blocks[-1].dstdata['label']['paper'] 
                loss= model_lp(positive_graph, negative_graph, blocks,  input_features, output_labels,train_mask,device,task_tag)
                        
                        
                loss1.append(loss.item())
                        #print (loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step() 
            
        #with tqdm.tqdm(dataloader_2 ) as tq:
        #    task_tag=3
        #    for step, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(tq):
        #        blocks = [b.to(device) for b in blocks] 
        #        positive_graph = positive_graph.to(device)
        #        negative_graph = negative_graph.to(device)
        #        input_features = blocks[0].srcdata['features']
        #        output_labels = blocks[-1].dstdata['label']['paper'] 
        #        loss= model_lp(positive_graph, negative_graph, blocks,  input_features, output_labels,train_mask,device,task_tag)
        #        loss1.append(loss.item())
                        #print (loss.item())
        #        opt.zero_grad()
        #        loss.backward()
        #        opt.step() 
        
            
    model_lp.eval()
    if device==torch.device('cuda:' + str(device_ids[0])):
        print ('aaaa') 
        #
     
      