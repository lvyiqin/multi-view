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
import run_aps 

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
  

#logits = model_lp.sage(hetero_graph.to(torch.device('cuda')), node_features)['paper']
 

class Model_LP(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_features_1, out_features, 
                 etypes):
        super().__init__()
        self.rgcn = StochasticThreeLayerRGCN(
            in_features, hidden_features, hidden_features_1, out_features, etypes)
        self.pred = ScorePredictor()

    def forward(self, positive_graph, negative_graph, blocks,  x,  output_labels,train_mask, device,task_tag):
        if task_tag==1:
            x0 = self.rgcn(blocks, x)[1]  
            dic_b_ref=dict()
            temp=0
            for i1 in blocks[2].ndata[dgl.NID]['ref'].tolist():
                dic_b_ref[i1]=temp
                temp=temp+1
            dic_b_paper=dict() #key是原来的节点标号，value是block的节点标号
            temp=0
            for i1 in blocks[2].ndata[dgl.NID]['paper'].tolist():
                dic_b_paper[i1]=temp
                temp=temp+1 
            dic_b_class=dict() #key是原来的节点标号，value是block的节点标号
            temp=0
            for i1 in blocks[2].ndata[dgl.NID]['class'].tolist():
                dic_b_class[i1]=temp
                temp=temp+1     
                
                
            in_ref=[]
            for i1 in positive_graph.ndata[dgl.NID]['ref'].tolist():
                in_ref.append( dic_b_ref[i1] )    
            in_ref=torch.tensor(in_ref).to(device)
            in_paper=[]
            for i1 in positive_graph.ndata[dgl.NID]['paper'].tolist():
                in_paper.append( dic_b_paper[i1] )    
            in_paper=torch.tensor(in_paper).to(device) #从positive graph中选取block[-1]的行
            in_class=[]
            for i1 in positive_graph.ndata[dgl.NID]['class'].tolist():
                in_class.append( dic_b_class[i1] )    
            in_class=torch.tensor(in_class).to(device)
 
            
            b_origin_node= blocks[-1].ndata[dgl.NID]['paper'].tolist() 
            b_dst_node= blocks[-1].dstnodes('paper').tolist() 
            #b_train_mask=[] 
            #for i1 in range(0,len(b_dst_node)):
            #    b_train_mask.append(  train_mask[b_origin_node[b_dst_node[i1]]].item() ) 
            #b_train_mask=torch.tensor(b_train_mask).to(device)
            
            

            
            x2=dict()
            x2['paper']=x0['paper'].index_select(0,in_paper)
            x2['ref']=x0['ref'].index_select(0,in_ref)
            x2['class']=x0['class'].index_select(0,in_class)
            
            pos_score = self.pred(positive_graph, x2) 
            neg_score = self.pred(negative_graph, x2)
            loss =  1*compute_loss(pos_score, neg_score)#+  F.cross_entropy(output_predictions[b_train_mask],output_labels[b_train_mask] )
                
        
        
        
        
        if   task_tag==2:            
            
            b_origin_node= blocks[-1].ndata[dgl.NID]['paper'].tolist() 
            b_dst_node= blocks[-1].dstnodes('paper').tolist() 
            b_train_mask=[] 
            for i1 in range(0,len(b_dst_node)):
                b_train_mask.append(  train_mask[b_origin_node[b_dst_node[i1]]].item() ) 
            b_train_mask=torch.tensor(b_train_mask).to(device)
            
            output_predictions = self.rgcn(blocks, x) [0]['paper'] 
            loss =   F.cross_entropy(output_predictions[b_train_mask],output_labels[b_train_mask] )
        
         
        if task_tag==3:
            x0 = self.rgcn(blocks, x)[1]  
            dic_b_ref=dict()
            temp=0
            for i1 in blocks[2].ndata[dgl.NID]['ref'].tolist():
                dic_b_ref[i1]=temp
                temp=temp+1
            dic_b_paper=dict() #key是原来的节点标号，value是block的节点标号
            temp=0
            for i1 in blocks[2].ndata[dgl.NID]['paper'].tolist():
                dic_b_paper[i1]=temp
                temp=temp+1 
            dic_b_class=dict() #key是原来的节点标号，value是block的节点标号
            temp=0
            for i1 in blocks[2].ndata[dgl.NID]['class'].tolist():
                dic_b_class[i1]=temp
                temp=temp+1     
                
                
            in_ref=[]
            for i1 in positive_graph.ndata[dgl.NID]['ref'].tolist():
                in_ref.append( dic_b_ref[i1] )    
            in_ref=torch.tensor(in_ref).to(device)
            in_paper=[]
            for i1 in positive_graph.ndata[dgl.NID]['paper'].tolist():
                in_paper.append( dic_b_paper[i1] )    
            in_paper=torch.tensor(in_paper).to(device) #从positive graph中选取block[-1]的行
            in_class=[]
            for i1 in positive_graph.ndata[dgl.NID]['class'].tolist():
                in_class.append( dic_b_class[i1] )    
            in_class=torch.tensor(in_class).to(device)
 
            
            b_origin_node= blocks[-1].ndata[dgl.NID]['paper'].tolist() 
            b_dst_node= blocks[-1].dstnodes('paper').tolist() 
            #b_train_mask=[] 
            #for i1 in range(0,len(b_dst_node)):
            #    b_train_mask.append(  train_mask[b_origin_node[b_dst_node[i1]]].item() ) 
            #b_train_mask=torch.tensor(b_train_mask).to(device)
            
            

            
            x2=dict()
            x2['paper']=x0['paper'].index_select(0,in_paper)
            x2['ref']=x0['ref'].index_select(0,in_ref)
            x2['class']=x0['class'].index_select(0,in_class)
            
            pos_score = self.pred(positive_graph, x2) 
            neg_score = self.pred(negative_graph, x2)
            loss =1*compute_loss(pos_score, neg_score) #+  F.cross_entropy(output_predictions[b_train_mask],output_labels[b_train_mask] )
                
            #x0 = self.rgcn(blocks, x)[0] 
            #pos_score_1 = self.pred(positive_graph, x0) 
            #neg_score_1 = self.pred(negative_graph, x0)
            #loss =  1*compute_loss(pos_score_1, neg_score_1)+ 1*compute_loss(pos_score, neg_score)
        
        
        return loss
    

 


os.environ['CUDA_VISIBLE_DEVICES']= '0,1,2,3'
os.environ['MASTER_ADDR']='localhost'
post = random.randint(1024 , 5000)
os.environ['MASTER_PORT'] = str(post)



world_size=1
rank=0
n = torch.cuda.device_count() // world_size
device_ids = list(range(rank * n, (rank + 1) * n))
torch.cuda.empty_cache() 

 
max_epoch=360 
acc_file_name=  r'clf_score_pnas_0LP1N5Tmul4_360.txt'
label_known_rate=0.05
train_rate=0.8
test_rate= 1
model_path=r'/home/xiezheng/Datasets/Sci_Papers/model_pnas_0LP1NC5Tmul4_360'
 
#with open(r'/home/xiezheng/Datasets/Sci_Papers/MAG/mag_papers_20_label_first.txt')as fl:
with open(r'/home/xiezheng/Datasets/Sci_Papers/PNAS/disp_label.txt')as fl:
#with open(r'/home/xiezheng/Datasets/Sci_Papers/APS/label_remove_new.txt')as fl:       
    lines = fl.readlines()

lines=lines[:int(test_rate*len(lines))]

num_paper = len(lines)
label_known_num=int(label_known_rate*num_paper) 
train_num=int(train_rate*num_paper)


 
 
 
  
    


labels = set(lines)
 
dic_labels = dict()
for i, x in enumerate(labels): #给标签编号
    dic_labels[x] = i 
    
node_origin_label =[]
for i in range(0,num_paper):
    node_origin_label.append(lines[i])

 

node_label = [] 
for i in range(0, len(node_origin_label)):
    node_label.append(dic_labels[node_origin_label[i]]) 
num_class = len(labels)
print('labels', len(labels), 'num_node', len(node_origin_label),label_known_rate) 
i = 0
node_classes = [[] for i in range(0, num_class)]  
for i in range(0,label_known_num):
    node_classes[dic_labels[node_origin_label[i]]].append(i)







i = 0
u_bine = np.random.randn(num_paper, 768)
#with open(r'/home/xiezheng/Datasets/Sci_Papers/MAG/mag_papers_20_BINE_bert.txt')as fo:
with open(r'/home/xiezheng/Datasets/Sci_Papers/PNAS/BINE.txt')as fo:        
#w#ith open(r'/home/xiezheng/Datasets/Sci_Papers/APS/APS_title_bert.txt')as fo:     
    lines = fo.readlines()

lines=lines[:int(test_rate*len(lines))]
for line in lines:
    line = line.strip().split(',')
    line[0] = line[0].replace('[', ' ').strip()
    line[-1] = line[-1].replace(']', ' ').strip()
    u_bine[i] = [float(x) for x in line]  #打乱后的新编号
    i = i + 1
i = 0
 
u_bine = torch.from_numpy(u_bine) 
u = preprocessing.scale(u_bine) 
u = torch.from_numpy(u)
u = u.to(torch.float32) 
 
s = []
e = []
#with open(r'/home/xiezheng/Datasets/Sci_Papers/MAG/mag_papers_20_uvw_first.txt')as fu:
with open(r'/home/xiezheng/Datasets/Sci_Papers/PNAS/pnas_uvw_new.txt')as fu:
#with open(r'/home/xiezheng/Datasets/Sci_Papers/APS/cite_uvw.txt')as fu:
    while 1:
        line = fu.readline()
        if line == '':
            break
        line = line.strip().split('\t')
        if int(line[0][2:])<num_paper:
            s.append(    int(line[0][2:]) )
            e.append(  int(line[1][2:]) )

dic = {}
for i, x in enumerate(list(set(e))):
    dic[x] = i
print(len(dic)) 
j = []
for i in range(len(e)):
    j.append(dic.get(e[i]))
rows = np.array(s)
cols = np.array(e)
v = np.ones(rows.shape[0])
 
W = sp.coo_matrix((v, (rows, cols)))
#WT = sp.coo_matrix((v, (cols, rows)))
#A = W.dot(WT)
#A = A.tocoo() 
A=W
rows=A.row 
cols=A.col 
      
#edge_list=[[],[]]
#for i in range(0,num_class): #
#        len_class_i=len(node_classes[i]) 
#        for j  in range (0 , len_class_i):       
#            edge_list[0].append(node_classes[i][j])
#            edge_list[1].append(i)
      
#构图
#etypes= [('paper', 'cit', 'ref' ),('paper','label','paper' )  ]
etypes= [('paper', 'cit', 'ref' ) ,('ref','cited','paper' ),('paper','cocit','paper' ),('paper','link','class' ) ]
data_dict = dict()
data_dict[etypes[0]]= (rows, cols)    #    cite_edges  
data_dict[etypes[1]]= (cols, rows)   
#data_dict[etypes[2]]= (row_1,col_1)     #    cocite_edges    
#data_dict[etypes[3]]= (edge_list[0],edge_list[1])   
G=dgl.heterograph(data_dict ) 
 

#添加图节点特征
hetero_graph=G
hetero_graph.nodes['paper'].data['features']=u  
paper_feats = hetero_graph.nodes['paper'].data['features'] 
cit_neigbor=[[] for i in range(0, hetero_graph.num_nodes('ref')   )] 
for i in range(0, len(rows)):
    cit_neigbor[cols[i]].append (rows[i])
zeros=torch.tensor(np.zeros(768))  
u_ref=[]
remove_nodes=[]
for  i in range(0, hetero_graph.num_nodes('ref')   ):
    if cit_neigbor[i]==[]:
        #u_ref.append(zeros)
        remove_nodes.append(i)
    else:     
        temp=u[cit_neigbor[i][0]]
        for j in range(1, len(cit_neigbor[i])):
            temp=u[cit_neigbor[i][j]]+temp 
        u_ref.append(temp /len(cit_neigbor[i]))
u_ref=torch.stack(u_ref) 

if remove_nodes!=[]:
    hetero_graph.remove_nodes(torch.tensor(remove_nodes), ntype='ref')
hetero_graph.nodes['ref'].data['features']=u_ref  

#hetero_graph=hetero_graph.to(torch.device('cuda'))   

node_features=hetero_graph.ndata['features'] 
n_features = u.shape[1] 

 
hetero_graph.nodes['paper'].data['label']=torch.tensor(node_label) 
labels = hetero_graph.nodes['paper'].data['label']




train_mask = [ True for i in range (0, label_known_num) ]
train_mask.extend(  [ False for i in range (label_known_num, num_paper) ]   )
train_mask=torch.Tensor(train_mask)
train_mask=train_mask.bool()
hetero_graph.nodes['paper'].data['train_mask']=train_mask
train_mask = hetero_graph.nodes['paper'].data['train_mask']

test_mask = [ False for i in range (0, train_num) ]
test_mask.extend(  [ True for i in range (train_num, num_paper) ]   )
test_mask=torch.Tensor(test_mask)
test_mask=test_mask.bool()
hetero_graph.nodes['paper'].data['test_mask']=test_mask
test_mask = hetero_graph.nodes['paper'].data['test_mask'] 
torch.distributed.init_process_group(backend='nccl',   init_method="env://",rank=0, world_size=1) 
           
if __name__ == '__main__':
    k = 5 
    hidden_feat_1=256
    
    acc_file_name= acc_file_name
    with open(acc_file_name, 'w') as fw:

        num_gpus =len(device_ids)
        for i in range(0,12):
            model_lp = Model_LP(n_features,512, hidden_feat_1, num_class, hetero_graph.etypes).to(torch.device('cuda'))      
            model_lp = DDP(model_lp, find_unused_parameters=True )     
            mp.spawn(run_aps.run, args=(list(range(num_gpus)),  model_lp,    hetero_graph ,   train_mask, max_epoch ),  nprocs=num_gpus, join=True )
            torch.save(model_lp.state_dict(),model_path+'_'+str(i)) 
            device='cpu'
            

            model_lp.eval()
                
            hetero_graph=hetero_graph.to('cpu')
            node_features=hetero_graph.ndata['features'] 
                
            model_lp=model_lp.to('cpu') 
            #
            node_features={key:node_features[key].to('cpu')  for key in node_features}
            acc = evaluate(model_lp   , hetero_graph , node_features , labels.to('cpu') , test_mask.to('cpu')  )
            print ('acc',acc) 
            torch.cuda.empty_cache()      
                    
            model_lp= model_lp.to('cpu') 
                    

            node_embeddings = model_lp.module.rgcn([hetero_graph,hetero_graph,hetero_graph], node_features )[1]['paper']
                #module.
                 
            z_u=node_embeddings


                    #z_u=u

            train_texts = z_u[0:label_known_num].cpu().detach().numpy()
            train_labels = node_label[0:label_known_num]
            test_texts = z_u[train_num:-1].cpu().detach().numpy()
            test_labels = node_label[train_num:-1]


            train_texts=  np.array( train_texts )
            train_labels=np.array( train_labels )
                    





                    # SGD
                    

            train_set = RayDMatrix(train_texts, train_labels)

            clf = RayXGBClassifier(
                    n_jobs=4,  # In XGBoost-Ray, n_jobs sets the number of actors
                    random_state=3
                )
                    
            clf.fit(train_texts, train_labels)

                    
            ff=clf.score(test_texts,test_labels)
            print(ff)
                    
            
            fw.write('%f\n' % acc) 
            fw.write('%f\n' % ff) 
                        
            fw.write('acc\n'    )