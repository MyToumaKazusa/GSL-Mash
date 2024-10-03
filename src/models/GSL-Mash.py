#mashup learning
import os.path
import numpy as np
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from typing import Any, Optional
from torch_geometric.nn import TransformerConv, Sequential, LayerNorm
from torch_geometric.nn.inits import glorot, zeros

from .base_learner import BaseLearner
from .metric import CosineSimilarity 
from .processor import KNNSparsify, NonLinearize, Symmetrize, Normalize
from .attention_learner import AttLearner
# from .gnn_learner import GNNLearner
# from .full_param_learner import FullParam
import torch.nn.init as init
import matplotlib.pyplot as plt



# from /SMRBmy/src/utils/metrics import calculate_precision, calculate_recall
from torchmetrics.classification.accuracy import Accuracy


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, use_dropout=False, dropout_p=0.5):
        super(ResidualBlock, self).__init__()
        self.use_dropout = use_dropout
        self.linear1 = nn.Linear(in_channels, mid_channels)
        self.batch_norm1 = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_p)
        self.linear2 = nn.Linear(mid_channels, out_channels)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        # 使用一个线性层来适配维度
        self.identity_layer = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        
        # 主路径
        out = self.linear1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.linear2(out)
        out = self.batch_norm2(out)
        
        # 适配维度的残差路径
        identity = self.identity_layer(identity)
        
        # 添加残差连接
        out += identity
        out = self.relu(out)
        return out


class GSL_MASH(LightningModule):
    r"""A MLP with two linear layers.

    Args:
        data_dir (str): Path to the folder where the data is located.
        api_embed_path (str): Path to embeddings vectors of api that is relative to :attr:`data_dir`.
        mlp_output_channels (int): Size of each output of the first linear layer.
        mashup_embed_channels (int): Size of each embedding vector of mashup.
        lr (float): Learning rate (default: :obj:`1e-3`).
        weight_decay (float): weight decay (default: :obj:`1e-5`).
    """
    def __init__(
        self,
        data_dir,
        api_embed_path: str,
        mlp_output_channels: int,
        mashup_embed_channels: int,
        lr: float ,
        weight_decay: float = 1e-5,
        #need to define
        gnn_layers = 10,
        api_graph=True,  # this should be set to true.
        api_out_channels =128,
        hidden_channels = 512,
        mashup_first_embed_channels = 768,#(1864x1536 and 256x300)
        mashup_new_embed_channels = 128,
        api_new_embed_channels = 128,
        heads = 1, 
        edge_message_agg='mean',
        edge_message = True,
        num_candidates = 932   #932 or 21495
    ):
        r"""A MLP with two linear layers.

        Args:
            data_dir (str): Path to the folder where the data is located.
            api_embed_path (str): Path to embeddings vectors of api that is relative to :attr:`data_dir`.
            mlp_output_channels (int): Size of each output of the first linear layer.
            mashup_embed_channels (int): Size of each embedding vector of mashup
            lr: Learning rate (default: :obj:`1e-3`).
            weight_decay: weight decay (default: :obj:`1e-5`).
        """
        super(GSL_MASH, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.register_buffer('api_embeds', torch.from_numpy(np.load(os.path.join(data_dir, api_embed_path))))
        self.num_api = self.api_embeds.size(0)

        self.api_out_channels = api_out_channels

        self.gnn_layers = gnn_layers     
        self.heads = heads
        self.mashup_first_embed_channels = mashup_first_embed_channels
        self.hidden_channels = hidden_channels
        self.edge_message = edge_message
        self.edge_message_agg = edge_message_agg
        self.mashup_new_embed_channels = mashup_new_embed_channels
        self.api_new_embed_channels = api_new_embed_channels
        self.mlp_output_channels = mlp_output_channels
        self.num_candidates = num_candidates

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.f1 = torchmetrics.F1Score(top_k=5)

        # I use direct path here, you need to change it
        self.mashup_path = "/home/liusihao/GSL-Mash/data/api_mashup/embeddings/text_bert_mashup_embeddings.npy"

        self.register_buffer('H', torch.zeros(self.num_api, self.api_out_channels))
        zeros(self.H) 
        
        self.file_path = 'qwq.txt'
        self.n = 0
        self._build_layers()



    def _build_layers(self):
        #######################################################################################################################################################
        #align_mlp
        self.align_mlp = nn.Sequential(
            nn.Linear(self.mashup_first_embed_channels, self.hidden_channels),#128
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, self.api_out_channels)#512
        )
        #api_reduce_mlp
        self.api_reduce_dem_mlp = nn.Sequential(
            nn.Linear(self.api_embeds.size(1), self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, self.api_out_channels)
        )
        #Hmlp
        self.Hmlp = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        #gnns
        gnn_layers = []
        for i in range(self.gnn_layers):
            gnn_layers.append(TransformerConv(self.api_out_channels,
                                              int(self.api_out_channels / self.heads), self.heads,
                                              edge_dim=self.api_out_channels if self.edge_message else None))
            gnn_layers.append(LayerNorm(self.api_out_channels))
        self.gnns = nn.ModuleList(gnn_layers)
        #唯一结构核心代码
        ##############################################################
        # 1 300 200 100
        # 2 300 250 150 100
        # 3 512 384 256 128 64
        #bn1 
        hid_chan = 512
        hid_chan2 = 384
        hid_chan3 = 256
        hid_chan4 = 128
        hid_chan5 = 64
        hid_chan6 = 32
        self.linear = nn.Sequential(
            ResidualBlock(self.mashup_new_embed_channels + self.api_new_embed_channels, hid_chan, hid_chan2),
            ResidualBlock(hid_chan2, hid_chan3, hid_chan4),
            ResidualBlock(hid_chan4, hid_chan5, hid_chan6),
            nn.Linear(hid_chan6, 1)
        )
        
        ##############################################################
        #GSL层
        metric = CosineSimilarity()
        processors = [KNNSparsify(450), NonLinearize(non_linearity='relu'), Symmetrize(), Normalize(mode='sym')]
        activation = nn.ReLU()
        self.gsl_learner = AttLearner(metric, processors, 10, 768, activation)

        processors2 = [KNNSparsify(10), NonLinearize(non_linearity='relu'), Symmetrize(), Normalize(mode='sym')]
        self.weight = 0.6
        self.mashup_learner = AttLearner(metric, processors2, 10, 768, activation) 
        #######################################################################################################################################################
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            print("yes")
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def training_step(self, batch: Any) -> STEP_OUTPUT:
        mashups_idx, labels = batch
        batch_size = mashups_idx.size(0)

        # mashups = self.align_mlp(mashups_idx) # -> [128]          [batch,128]
        mashups = self.mashup_learning(mashups_idx)
        api_ori = self.api_reduce_dem_mlp(self.api_embeds) #  [768] -> [128]    [932,128]

        #######################################
        edge_index, edge_attr = self.format_samples(mashups, labels)  # mashup需要是[B,C],label需要是[B,N]

        self.edge = edge_index
        # self.draw_plt(edge_index, name="start")

        #################dropout or gsl###########################
        adj = self.gsl_learner(self.api_embeds)

        # self.draw_plt(adj, name="mid")
        edge_index1 , edge_attr1 , empt= self.mix_bing(adj, edge_index, edge_attr)

        # self.draw_plt(edge_index1, name="end")

        if empt == 1:
            edge_index = edge_index1
            edge_attr = edge_attr1
        ###########################################################

        edge_index = edge_index.cuda()
        edge_attr = edge_attr.cuda()
        
        for i in range(self.gnn_layers):
            apis = self.gnns[i * 2](api_ori, edge_index, edge_attr)
            apis = self.gnns[i * 2 + 1](apis)

        ########################################

        Hid = torch.cat((apis, self.H), dim=-1).requires_grad_()
        Hid = torch.cat((Hid, api_ori), dim=-1).requires_grad_()
        Hid = self.Hmlp(Hid) #[932,256] - [932,128]

        self.H = Hid.detach()
        apis = Hid

        mashups_fin = mashups.unsqueeze(1).repeat(1, self.num_api, 1)#[batch,932,128]
        apis_fin = apis.unsqueeze(0).repeat(batch_size, 1, 1)#[batch,932,128]
        input_feature = torch.cat((mashups_fin, apis_fin), dim=-1).requires_grad_()#[batch,932,256]4

        input_feature = input_feature.view(batch_size * self.num_candidates, 256)

        preds = self.linear(input_feature).requires_grad_()

        preds = preds.view(batch_size, self.num_candidates, -1)

        preds = preds.view(batch_size, self.num_api)
        loss = self.criterion(preds, labels.float())
        
        if torch.isnan(loss):
            with open(self.file_path, 'a') as file:
                file.write(f' {loss}\n')
                file.write(f' {preds}\n')
                file.write(f' {labels}\n')
        
        self.log('train/loss', loss)

        return loss

    
    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups_idx, labels = batch
        batch_size = mashups_idx.size(0)

        # mashups = self.align_mlp(mashups_idx) # -> [128]          [batch,128]
        mashups = self.mashup_learning(mashups_idx)
        
        # edge_index, edge_attr = self.format_samples(mashups, labels)  
        
        Hid = self.H
        apis = Hid

        mashups_fin = mashups.unsqueeze(1).repeat(1, self.num_api, 1)#[batch,932,128]
        apis_fin = apis.unsqueeze(0).repeat(batch_size, 1, 1)#[batch,932,128]
        input_feature = torch.cat((mashups_fin, apis_fin), dim=-1).requires_grad_()#[batch,932,256]4

        input_feature = input_feature.view(batch_size * self.num_candidates, 256)

        preds = self.linear(input_feature).requires_grad_()

        preds = preds.view(batch_size, self.num_candidates, -1)

        preds = preds.view(batch_size, self.num_api)

        # self.update_H(edge_index, edge_attr)

        self.log('val/F1', self.f1(preds, labels), on_step=False, on_epoch=True, prog_bar=False)
        


    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mashups_idx, labels = batch
        batch_size = mashups_idx.size(0)

        # mashups = self.align_mlp(mashups_idx) # -> [128]          [batch,128]
        mashups = self.mashup_learning(mashups_idx)

        # edge_index, edge_attr = self.format_samples(mashups, labels)  
        
        Hid = self.H
        apis = Hid

        mashups_fin = mashups.unsqueeze(1).repeat(1, self.num_api, 1)#[batch,932,128]
        apis_fin = apis.unsqueeze(0).repeat(batch_size, 1, 1)#[batch,932,128]
        input_feature = torch.cat((mashups_fin, apis_fin), dim=-1).requires_grad_()#[batch,932,256]4

        input_feature = input_feature.view(batch_size * self.num_candidates, 256)

        preds = self.linear(input_feature).requires_grad_()

        preds = preds.view(batch_size, self.num_candidates, -1)

        preds = preds.view(batch_size, self.num_api)

        # self.update_H(edge_index, edge_attr)

        return {
            'preds': preds,
            'targets': labels
        }

    def configure_optimizers(self):
        optimizer_model = torch.optim.Adam(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer_model
        
    def update_H(self, edge_index, edge_attr):
        
        # edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=dropout, force_undirected=True)
        adj = self.gsl_learner(self.api_embeds)
        edge_index , edge_attr , notempty = self.mix_bing(adj, edge_index, edge_attr)
        if notempty == 0:
            return 
        edge_index = edge_index.cuda()
        edge_attr = edge_attr.cuda()
        api_ori = self.api_reduce_dem_mlp(self.api_embeds)

        for i in range(self.gnn_layers):
            api = self.gnns[i * 2](api_ori, edge_index, edge_attr )
            api = self.gnns[i * 2 + 1](api)
        
        Hid = torch.cat((api, self.H), dim=-1)
        Hid = torch.cat((Hid, api_ori), dim=-1)
        self.H = self.Hmlp(Hid)
        

    #self.format_samples(Xs, Ys)  from Batch
    def format_samples(self, Xs, Ys, **kwargs):
        """
        Xs: [B, C]  # C: message channel
        Ys: [B, N]  # N: API numbers
        """
        edge_message = {}
        for x, y in zip(Xs, Ys):
            invoked_apis = y.nonzero(as_tuple=True)[0]  #形如选坐标之后只保留第一维
            invoked_apis = invoked_apis.cpu().numpy()
            for api_i in invoked_apis:
                for api_j in invoked_apis:
                    edge_message[(api_i, api_j)] = edge_message.get((api_i, api_j), []) + [x]
        src, dst, edge_attr = [], [], []
        for (u, v), messages in edge_message.items():
            #会有从自己到自己的边
            src.append(u)
            dst.append(v)
            if self.edge_message_agg == 'mean':
                edge_attr.append(torch.mean(torch.stack(messages, dim=0), dim=0))#会保留原信息，增加某一维度的数量
            else:
                edge_attr.append(torch.sum(torch.stack(messages, dim=0), dim=0))
        src = torch.tensor(src)#这里是变成张量不是编码，数据不会有形式上的改变
        dsc = torch.tensor(dst)
        #torch.stack放在一起 [B]+[B]=[2,B]
        edge_index = torch.stack([src, dsc], dim=0)#边索引，第一行是起点，第二行是终点
        edge_index = edge_index.to(self.device)
        edge_attr = torch.stack(edge_attr, dim=0).type_as(Xs)#是一个包含所有边特征的张量
        return edge_index, edge_attr            #？？？edge_attr指的是边的标签信息聚合还是节点的连接信息


    def mix_bing(self, adj, edge_index, edge_attr):
        notempty = 0
        filtered_edge_index = [[], []]
        filtered_edge_attr = []
        for i in range(edge_index.size(1)):
            src = edge_index[0][i]
            dst = edge_index[1][i]
            if adj[src, dst] != 0:
                continue
                # 若 [src, dst] 在 adj 中不为 0，则将对应的边属性乘以 adj 中的值，并保留
                filtered_edge_index[0].append(src)
                filtered_edge_index[1].append(dst)
                filtered_edge_attr.append(edge_attr[i] * 0.0001) ######!!!

            else :
                notempty = 1
                filtered_edge_index[0].append(src)
                filtered_edge_index[1].append(dst)
                filtered_edge_attr.append(edge_attr[i])
            
        if notempty == 0 :
            return torch.tensor(filtered_edge_index), filtered_edge_attr, notempty
        filtered_edge_index = torch.tensor(filtered_edge_index)
        filtered_edge_attr = torch.stack(filtered_edge_attr)
        return filtered_edge_index, filtered_edge_attr, notempty

    def mashup_learning(self, mashup_idx):
        
        torch.set_printoptions(profile="full")
        np.set_printoptions(threshold=np.inf)
        
        mashups = np.load(self.mashup_path) 
        mashups = torch.from_numpy(mashups).cuda()
        adj_matrix = self.mashup_learner(mashups)

        # 确保邻接矩阵的对角线为0（一个节点与自身的相似度不计入）
        eye = torch.eye(adj_matrix.size(0), device=adj_matrix.device)
        adj_matrix = adj_matrix * (1 - eye)
        # 计算邻接矩阵每行的和，用于归一化
        row_sums = adj_matrix.sum(dim=1, keepdim=True)
        # 防止除以零
        row_sums[row_sums == 0] = 1
        # 归一化邻接矩阵，使得每行的和为(1 - weight)
        normalized_adj_matrix = adj_matrix / row_sums * (1 - self.weight)
        mashups = self.weight * mashups + torch.matmul(normalized_adj_matrix, mashups)
        mashups = self.align_mlp(mashups[mashup_idx])
        return mashups
    
    def draw_plt(self, index, name):
        if name == 'start':
            self.flag = 1
            index_t = index.cpu()
            index_np = index_t.numpy()

            # 分别获取源和目的节点
            src_nodes = index_np[0, :]
            dst_nodes = index_np[1, :]

            # 获取所有唯一的节点，并排序
            unique_nodes = np.unique(np.concatenate((src_nodes, dst_nodes)))
            sorted_nodes = np.sort(unique_nodes)
            self.sorted = sorted_nodes
            # 构造节点到新索引的映射
            node_to_index = {node: idx for idx, node in enumerate(sorted_nodes)}

            # 构建新的邻接矩阵
            n = len(sorted_nodes)  # 新矩阵的大小
            if(n > self.n):
                self.n = n
            else:
                self.flag = 0
                return
            adj_matrix = np.zeros((n, n), dtype=int)

            # 填充邻接矩阵
            for src, dst in zip(src_nodes, dst_nodes):
                i, j = node_to_index[src], node_to_index[dst]
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  # 确保对称性
            plt.spy(adj_matrix)
            plt.savefig('start.png', dpi=500,bbox_inches = 'tight')
            plt.show()
        if name == 'mid':
            if(self.flag == 0):
                return
            adj = index.cpu()
            adj = adj.detach().numpy()
            sparse_matrix = self.edge.cpu()
            sparse_matrix = sparse_matrix.numpy()
            # 提取稀疏矩阵中的所有唯一数字并排序
            unique_numbers = np.unique(sparse_matrix)
            sorted_numbers = np.sort(unique_numbers)
            
            # 创建映射关系从原始数字到新的索引
            index_map = {number: index for index, number in enumerate(sorted_numbers)}
            
            # 初始化新的稠密矩阵，大小为 n x n
            n = len(sorted_numbers)
            new_matrix = np.zeros((n, n), dtype=adj.dtype)
            
            # 遍历原 adj 矩阵中的相关行和列，复制数据到新矩阵
            for i, src in enumerate(sorted_numbers):
                for j, dst in enumerate(sorted_numbers):
                    if src-1 < adj.shape[0] and dst-1 < adj.shape[1]:  # 确保索引有效
                        new_matrix[i, j] = adj[src-1, dst-1]
            plt.spy(new_matrix)
            plt.savefig('mid.png', dpi=500,bbox_inches = 'tight')
            plt.show()
        if name == 'end' :
            if(self.flag == 0):
                return
            index_t = index.cpu()
            index_np = index_t.numpy()


            sorted_nodes = self.sorted
            node_to_index = {node: idx for idx, node in enumerate(sorted_nodes)}

            # 初始化新的稠密矩阵
            n = len(sorted_nodes)
            dense_matrix = np.zeros((n, n), dtype=int)

            # 填充矩阵
            for src, dst in zip(index_np[0], index_np[1]):
                # 获取节点对应的新索引
                src_index = node_to_index[src]
                dst_index = node_to_index[dst]
                # 用1填充对应的位置
                dense_matrix[src_index, dst_index] = 1
                dense_matrix[dst_index, src_index] = 1  # 确保对称性

            plt.spy(dense_matrix)
            plt.savefig('end.png', dpi=500,bbox_inches = 'tight')
            plt.show()