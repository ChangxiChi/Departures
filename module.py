import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch.nn import LayerNorm
from nn import timestep_embedding
from torch_geometric.nn import GATConv
from torch_geometric.data import Batch,Data
import rff
# pip install random-fourier-features-pytorch

# based on v2
# graph model

class MLP(nn.Module):
    def __init__(self,
                 sizes,
                 batch_norm=True,
                 last_layer_act='linear',
                 append_layer_width=None,
                 append_layer_position=None,
                 act="SiLU"
                 ):
        super(MLP, self).__init__()
        self.batch_norm = batch_norm
        self.layers = []

        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

            if batch_norm and i < len(sizes) - 2:
                self.layers.append(nn.BatchNorm1d(sizes[i + 1]))

            if i < len(sizes) - 2 and act == "ReLU":
                self.layers.append(nn.ReLU())

            if i < len(sizes) - 2 and act == "SiLU":
                self.layers.append(nn.SiLU())


        if append_layer_width is not None:
            if append_layer_position == "first":
                self.layers.insert(0, nn.Linear(sizes[0], append_layer_width))
                self.layers.insert(1, nn.BatchNorm1d(append_layer_width))
                self.layers.insert(2, nn.ReLU())
            elif append_layer_position == "last":
                self.layers.append(nn.Linear(sizes[-1], append_layer_width))
                self.layers.append(nn.BatchNorm1d(append_layer_width))
                self.layers.append(nn.ReLU())



        if last_layer_act == "ReLU":
            self.layers.append(nn.ReLU())
        elif last_layer_act == "linear":
            self.layers.append(nn.Identity())
        elif last_layer_act == "LeakyReLU":
            self.layers.append(nn.LeakyReLU(negative_slope=0.01))

        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        is_single = False

        if x.dim() == 3:
            b, l, d = x.shape
            if b == 1 and self.batch_norm:
                is_single = True
                x = x.repeat(2, 1, 1)
                b = 2
            x = x.view(b * l, d)
            x = self.network(x)
            x = x.view(b, l, -1)
            if is_single:
                x = x[0:1]
        else:
            if x.shape[0] == 1 and self.batch_norm:
                is_single = True
                x = x.repeat(2, 1)
            x = self.network(x)
            if is_single:
                x = x[0:1]

        return x


class Block(nn.Module):
    def __init__(self,
                 gene_num,
                 output_dim,
                 time_embed_dim,
                 cell_type_embed_dim,
                 depth=3,
                 ):
        super(Block, self).__init__()
        '''
        time_embed_dim equals to cell_type_embed_dim
        '''
        # self.mlp=MLP([gene_init_dim+1]+2*[2*hidden_dim]+[output_dim])
        self.mlp1 = MLP([gene_num] + depth*[output_dim])
        self.time_encoder = MLP([time_embed_dim]+2*[output_dim])
        self.cell_type_encoder=MLP([cell_type_embed_dim]+depth*[output_dim])
        self.mlp2 = MLP((depth+1)*[output_dim])
        self.mlp3 = MLP((depth+1)*[output_dim])

    def forward(self,x_l,time_emb,cell_type=None):
        '''
        :param x_l: input, shape:[batch_size,gene_num]
        '''
        f=self.mlp1(x_l)
        # f = th.cat([f , self.time_encoder(time_emb)],dim=1)
        f=self.mlp2(f+self.time_encoder(time_emb))
        if cell_type is not None:
            f = f + self.cell_type_encoder(cell_type)
        f=self.mlp3(f)
        return f


class MultiLayerGAT(nn.Module):
    def __init__(self, dim, heads=[4, 4], dropout=0.1, use_residual=False, use_norm=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.heads = heads
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_norm = use_norm

        self.norms = nn.ModuleList()

        for i in range(len(dim) - 1):
            in_dim = dim[i]
            out_dim = dim[i + 1] // heads[i]
            self.layers.append(GATConv(in_dim, out_dim, heads=heads[i], concat=True))
            if self.use_norm:
                self.norms.append(LayerNorm(dim[i + 1]))


    def forward(self, g, feature):  # feature: [B, N, D]
        B, N, D = feature.shape
        x = feature.reshape(B * N, D)  # Flattening the feature

        # Create the batched edge_index using PyG's Batch
        edge_index = g.edge_index  # [2, E]

        # Batch the graph data using Batch.from_data_list
        data_list = []
        for b in range(B):
            data = Data(x=x[b * N: (b + 1) * N], edge_index=edge_index)
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        x = batch.x  # [B*N, D]
        edge_index = batch.edge_index  # [2, E]

        for i, layer in enumerate(self.layers):
            h = layer(x, edge_index)  # [B*N, hidden_dim]

            if self.use_residual:
                if h.shape == x.shape:
                    h = h + x  # Add residual connection if the dimensions match
            if self.use_norm:
                h = self.norms[i](h)

            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            x = h

        x = x.view(B, N, -1)  # Reshaping the result back to [B, N, hidden_dim]
        return x


class Dosager(nn.Module):
    def __init__(self,
                 dim,
                 ):
        """
        The `dosager` module combines molecular representations (`x`) and
            dosage information (`dosage`) to model the effect of dosage on the molecular response.
        """
        super(Dosager, self).__init__()
        self.mlp=MLP(sizes=[dim+1,dim,dim,1])

    def forward(self, x, dosage):
        x_dosage = th.cat((x, dosage), dim=1)
        return self.mlp(x_dosage).sigmoid()


class Mole_encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 ):
        """
        :param input_dim: dimension of input molecular features extracted by Uni-Mol
        :param output_dim: dimension of output molecular features, equal to gene embedding dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dosager = Dosager(dim=output_dim)
        self.mole_mlp = MLP(sizes=[input_dim,input_dim,output_dim])

    def forward(self, mole_embedding, dosage):
        """
        :param
         mole_embedding: molecule embedding extracted by pretrained Uni-Mol
         dosage: dosage of molecular drug
        """
        # mole_embedding shape [b,output_dim]
        mole_embedding = self.mole_mlp(mole_embedding)

        # dosage_embed shape [b,1]
        dosage_embed = self.dosager(mole_embedding, dosage)

        # mole_dosage_embed shape [b,output_dim]
        mole_dosage_embed = dosage_embed*mole_embedding

        return mole_dosage_embed


class GRN_conditional_network(nn.Module):
    def __init__(self,
                 gene_num,
                 gene_init_dim,
                 mole_dim,
                 output_dim,
                 GRN,
                 cell_type_embed_dim,
                 cell_type_num,
                 time_pos_dim,
                 pert_type="gene",
                 ):
        super(GRN_conditional_network, self).__init__()
        self.gene_num = gene_num
        self.gene_init_dim = gene_init_dim
        self.pert_type = pert_type

        self.grn=GRN
        self.gene_embedding=nn.Embedding(gene_num,gene_init_dim)
        self.indv_w = nn.Parameter(th.rand(1,gene_num, gene_init_dim))


        nn.init.xavier_normal_(self.indv_w)
        # nn.init.xavier_normal_(self.indv_b)

        if cell_type_num >1:
            self.cell_type_encoder=MLP(sizes=[cell_type_embed_dim]+2*[gene_init_dim])

        self.time_pos_dim=time_pos_dim
        self.time_encoder=MLP(sizes=[time_pos_dim]+2*[gene_init_dim])

        self.gene_encoder=MLP(sizes=4*[gene_init_dim])

        self.encoding = rff.layers.PositionalEncoding(sigma=1.0, m=gene_init_dim // 2)

        self.ln_1 = nn.LayerNorm(gene_init_dim)

        self.gnn_1=MultiLayerGAT(dim=3*[gene_init_dim],heads=[2,2])

        if self.pert_type == "molecular":
            self.mlp=MLP(sizes=3*[gene_init_dim])
            self.mole_encoder = Mole_encoder(input_dim=mole_dim, output_dim=gene_init_dim)
            self.gene_mole_encoder=MLP(sizes=[2*gene_init_dim] + 2*[gene_init_dim])
            # self.gnn_2=MultiLayerGAT(dim=3*[gene_init_dim],heads=[2,2])

            # self.encoder=Block(gene_num=gene_num,
            #         output_dim=gene_num,
            #         time_embed_dim=time_pos_dim,
            #         cell_type_embed_dim=cell_type_embed_dim)
            
            # self.decoder=MLP(sizes=[gene_num]+2*[gene_init_dim]+[gene_num])
        
        else:
            self.mlp=MLP(sizes=3*[gene_num])
            # self.decoder=MLP(sizes=3*[gene_num])


    def forward(self,x_t=None,x_0=None,time_emb =None,cell_type_emb=None,mole=None,dosage=None,knockout=None,single=True):
        self.grn.edge_index = self.grn.edge_index.to(x_t.device)
        
        
        if self.pert_type == "gene":
            gene_initialization = ( 
                        self.ln_1(self.gene_embedding.weight.unsqueeze(0).repeat(time_emb.shape[0], 1, 1))
                        +self.ln_1(self.encoding(x_t.unsqueeze(-1)))
                        +self.ln_1(self.time_encoder(time_emb))
                        )
            if cell_type_emb is not None:
                gene_initialization = gene_initialization + self.ln_1(self.cell_type_encoder(cell_type_emb).unsqueeze(1))
            """
            mask corresponding gene node feature
            """
            if single is True:
                mask = th.ones(len(knockout), self.gene_num, self.gene_init_dim).to(x_t.device)
                mask.scatter_(1, knockout.view(-1, 1, 1).expand(-1, -1, self.gene_init_dim), 0)
                masked_gene_initialization = gene_initialization * mask
                res = self.gnn_1(feature=masked_gene_initialization, g=self.grn)  # shape: [b,gene_num,output_dim]
                res = self.gene_encoder(res)
                gwf = (res * self.indv_w.squeeze(-1)).sum(dim=-1, keepdim=True)
                gwf = gwf.view(time_emb.shape[0],-1)
                gwf = gwf.squeeze()

                # h=self.encoder(x_t,
                #                time_emb.squeeze(),
                #                cell_type=cell_type_emb)
                
                # f=self.mlp(h+gwf)
                # output=self.decoder(f+x_t)

                # return output
                return gwf + x_0

            else:
                mask = th.ones(len(knockout), self.gene_num, self.gene_init_dim).to(x_t.device)
                expanded_idx = knockout.unsqueeze(-1).expand(-1, -1, self.gene_init_dim)
                mask.scatter_(1, expanded_idx, 0)
                masked_gene_initialization = gene_initialization * mask
                res = self.gnn_1(feature=masked_gene_initialization, g=self.grn)

        # elif self.pert_type == "molecular":
        #     gene_initialization = ( 
        #                         self.ln_1(self.gene_embedding.weight.unsqueeze(0).repeat(time_emb.shape[0], 1, 1))
        #                        +self.ln_1(self.encoding(x_t.unsqueeze(-1)))
        #                        +self.ln_1(self.time_encoder(time_emb))
        #                        )
            
        #     if len(dosage.shape)!=2:
        #         dosage = dosage.unsqueeze(-1)
        #     mole_dosage_emb = self.ln_1(self.mole_encoder(mole, dosage))
        #     gene_initialization = self.ln_1(gene_initialization) + mole_dosage_emb.unsqueeze(1).expand(-1, self.gene_num, -1)
        #     res = self.gnn_1(feature=gene_initialization, g=self.grn) # shape: [b,gene_num,output_dim]

        #     if cell_type_emb is not None:
        #         res = self.ln_1(res) + self.ln_1(self.cell_type_encoder(cell_type_emb).unsqueeze(1))
            
        #     res = res+self.encoding(x_t.unsqueeze(-1))
        #     res = self.gnn_2(feature=res, g=self.grn)

        #     res=self.gene_encoder(res)
        #     gwf=(res * self.indv_w.squeeze(-1)).sum(dim=-1, keepdim=True)
        #     gwf=gwf.view(time_emb.shape[0],-1)
        #     h=gwf.squeeze()

        #     return h+x_0

        elif self.pert_type == "molecular":
            gene_initialization = ( 
                                self.ln_1(self.gene_embedding.weight.unsqueeze(0).repeat(time_emb.shape[0], 1, 1))
                            +self.ln_1(self.encoding(x_t.unsqueeze(-1)))
                            +self.ln_1(self.time_encoder(time_emb))
                            )
            
            if cell_type_emb is not None:
                gene_initialization = gene_initialization + self.ln_1(self.cell_type_encoder(cell_type_emb).unsqueeze(1))

            if len(dosage.shape)!=2:
                dosage = dosage.unsqueeze(-1)
            mole_dosage_emb = self.ln_1(self.mole_encoder(mole, dosage))
            gene_initialization = self.ln_1(self.mlp(gene_initialization))
            # gene_initialization = th.cat([gene_initialization, mole_dosage_emb.unsqueeze(1).expand(-1, self.gene_num, -1)],dim=-1)
            # gene_initialization = self.gene_mole_encoder(gene_initialization)
            gene_initialization = gene_initialization + mole_dosage_emb.unsqueeze(1).expand(-1, self.gene_num, -1)
            res = self.gnn_1(feature=gene_initialization, g=self.grn) # shape: [b,gene_num,output_dim]

            # if cell_type_emb is not None:
            #     res = res + self.ln_1(self.cell_type_encoder(cell_type_emb).unsqueeze(1))
            # res=res+self.ln_1(self.time_encoder(time_emb))

            res=self.gene_encoder(res)
            gwf=(res * self.indv_w.squeeze(-1)).sum(dim=-1, keepdim=True)
            gwf=gwf.view(time_emb.shape[0],-1)
            f=gwf.squeeze()
            
            return f+x_0
            
        else:
            return None

        
class model(nn.Module):
    def __init__(self,
                 gene_num,
                 GRN,
                 gene_init_dim=64,
                 hidden_dim=512,
                 time_pos_dim=256,
                 time_embed_dim=256,
                 gene_wise_embed_dim=256,
                 cell_type_num=None,
                 cell_type_embed_dim=256,
                 data_name="adamson",
                 mole_dim=512,
                 pert_type="gene",
                 ):
        super(model,self).__init__()
        self.gene_num = gene_num
        self.data_name = data_name
        self.time_pos_dim = time_pos_dim
        self.time_embed_dim = time_embed_dim
        self.cell_type_num = cell_type_num # number of cell type

        if cell_type_num >1:
            self.cell_type_embedding = nn.Embedding(cell_type_num, cell_type_embed_dim)
        else:
            self.cell_type_embedding = None

        self.control_net=GRN_conditional_network(gene_num=gene_num,
                                                 gene_init_dim=gene_init_dim,
                                                 mole_dim=mole_dim,
                                                 output_dim=gene_wise_embed_dim,
                                                 GRN=GRN,
                                                 time_pos_dim=time_embed_dim,
                                                 cell_type_num=cell_type_num,
                                                 cell_type_embed_dim=cell_type_embed_dim,
                                                 pert_type=pert_type,
                                                 )


    def forward(self, x_t, timesteps, x_0=None,knockout=None, cell_type=None, mole=None, dosage=None, single=True):
        cond_info_cond = self.control_net(
                x_t=x_t,
                x_0=x_0,
                time_emb=timestep_embedding(timesteps, self.time_pos_dim),
                cell_type_emb=self.cell_type_embedding(cell_type.view(-1)) if self.cell_type_embedding is not None else None,
                mole=mole if mole is not None else None,
                dosage=dosage if dosage is not None else None,
                knockout=knockout if knockout is not None else None,
                single=single,
            )

        return cond_info_cond
    

class GRN_conditional_network_discrete(nn.Module):
    def __init__(self,
                 gene_num,
                 gene_init_dim,
                 mole_dim,
                 output_dim,
                 GRN,
                 cell_type_embed_dim,
                 cell_type_num,
                 time_pos_dim,
                 pert_type="gene",
                 ):
        super(GRN_conditional_network_discrete, self).__init__()
        self.gene_num = gene_num
        self.gene_init_dim = gene_init_dim
        self.pert_type = pert_type

        self.grn=GRN
        self.gene_embedding=nn.Embedding(gene_num,gene_init_dim)
        self.label_embedding=nn.Embedding(2,gene_init_dim)
        self.indv_w = nn.Parameter(th.rand(1,gene_num, gene_init_dim))
        nn.init.xavier_normal_(self.indv_w)
        # nn.init.xavier_normal_(self.indv_b)

        if cell_type_num >1:
            self.cell_type_encoder=MLP(sizes=[cell_type_embed_dim]+2*[gene_init_dim])

        self.time_pos_dim=time_pos_dim
        self.time_encoder=MLP(sizes=[time_pos_dim]+2*[gene_init_dim],act="ReLU")

        self.mlp_1=MLP(sizes=3*[gene_init_dim])
        # self.mlp_2=MLP(sizes=4*[gene_init_dim])
        # self.mlp_3=MLP(sizes=4*[gene_init_dim]+[output_dim])

        self.gnn_1=MultiLayerGAT(dim=3*[gene_init_dim],heads=[2,2])
        self.gnn_2=MultiLayerGAT(dim=3*[gene_init_dim],heads=[2,2])

        self.ln_1 = nn.LayerNorm(gene_init_dim)

        if self.pert_type == "molecular":
            self.mole_encoder = Mole_encoder(input_dim=mole_dim, output_dim=gene_init_dim)
            self.gene_mole_encoder=MLP(sizes=[2*gene_init_dim] + 2*[gene_init_dim])
        
        self.decoder=MLP(sizes=4*[gene_num])


    def forward(self, x_t_d = None, x_0=None, time_emb = None, cell_type_emb = None, mole = None, dosage = None, knockout = None, single = True):
        self.grn.edge_index = self.grn.edge_index.to(x_t_d.device)
        gene_initialization = self.gene_embedding.weight # shape: [gene_num,d]
        gene_initialization = (gene_initialization.unsqueeze(0).repeat(time_emb.shape[0], 1, 1)
                               +self.label_embedding(x_t_d.long())
                               +self.time_encoder(time_emb))
        # gene_initialization = (gene_initialization.unsqueeze(0).repeat(time_emb.shape[0], 1, 1)
        #                        +self.time_encoder(time_emb))

        if cell_type_emb is not None:
            gene_initialization = gene_initialization + self.cell_type_encoder(cell_type_emb).unsqueeze(1)

        if self.pert_type == "gene":
            """
            mask corresponding gene node feature
            """
            if single is True:
                mask = th.ones(len(knockout), self.gene_num, self.gene_init_dim).to(x_t_d.device)
                mask.scatter_(1, knockout.view(-1, 1, 1).expand(-1, -1, self.gene_init_dim), 0)
                masked_gene_initialization = gene_initialization * mask
                h = self.gnn_1(feature=masked_gene_initialization, g=self.grn)  # shape: [b,gene_num,output_dim]
                h = self.ln_1(h) + self.ln_1(self.label_embedding(x_t_d.long())) * mask
                res = self.gnn_2(feature=h, g=self.grn)
            else:
                mask = th.ones(len(knockout), self.gene_num, self.gene_init_dim).to(x_t_d.device)
                expanded_idx = knockout.unsqueeze(-1).expand(-1, -1, self.gene_init_dim)
                mask.scatter_(1, expanded_idx, 0)
                masked_gene_initialization = gene_initialization * mask
                h = self.gnn_1(feature=masked_gene_initialization, g=self.grn)
                res = self.gnn_2(feature=h+self.label_embedding(x_t_d.long()), g=self.grn)
            
            res=self.mlp_1(res)
            gwf = (res * self.indv_w.squeeze(-1)).sum(dim=-1, keepdim=True)
            gwf=gwf.view(time_emb.shape[0],-1)
            # h = self.decoder(gwf + x_0)

            return gwf

        elif self.pert_type == "molecular":
            if len(dosage.shape)!=2:
                dosage = dosage.unsqueeze(-1)
            mole_dosage_emb = self.mole_encoder(mole, dosage)

            # gene_initialization=self.gene_mole_encoder(th.cat([gene_initialization, mole_dosage_emb.unsqueeze(1).expand(-1, self.gene_num, -1)],dim=2))
            
            h = self.gnn_1(feature=gene_initialization, g=self.grn) # shape: [b,gene_num,output_dim]
            h = h + self.label_embedding(x_t_d.long()) + mole_dosage_emb.unsqueeze(1).expand(-1, self.gene_num, -1)
            res = self.gnn_2(feature=h, g=self.grn)
            res = self.mlp_1(h)
            gwf = (res * self.indv_w.squeeze(-1)).sum(dim=-1, keepdim=True)
            gwf = gwf.view(time_emb.shape[0],-1)
            # h = self.decoder(gwf + x_0)

            return gwf
            
        else:
            return None


class model_discrete(nn.Module):
    def __init__(self,
                 gene_num,
                 GRN,
                 gene_init_dim=64,
                 time_pos_dim=256,
                 time_embed_dim=256,
                 gene_wise_embed_dim=256,
                 cell_type_num=None,
                 cell_type_embed_dim=256,
                 data_name="adamson",
                 mole_dim=512,
                 pert_type="gene",
                 ):
        super(model_discrete,self).__init__()
        self.gene_num = gene_num
        self.data_name = data_name
        self.time_pos_dim = time_pos_dim
        self.time_embed_dim = time_embed_dim
        self.cell_type_num = cell_type_num # number of cell type

        if cell_type_num >1:
            self.cell_type_embedding = nn.Embedding(cell_type_num, cell_type_embed_dim)
        else:
            self.cell_type_embedding = None

        self.control_net=GRN_conditional_network_discrete(gene_num=gene_num,
                                                        gene_init_dim=gene_init_dim,
                                                        mole_dim=mole_dim,
                                                        output_dim=gene_wise_embed_dim,
                                                        GRN=GRN,
                                                        time_pos_dim=time_embed_dim,
                                                        cell_type_num=cell_type_num,
                                                        cell_type_embed_dim=cell_type_embed_dim,
                                                        pert_type=pert_type,
                                                        )

    def forward(self, x_t_d, timesteps, x_0=None,knockout=None, cell_type=None, mole=None, dosage=None, single=True):
        h = self.control_net(
                x_t_d=x_t_d,
                x_0=x_0,
                time_emb=timestep_embedding(timesteps, self.time_pos_dim),
                cell_type_emb=self.cell_type_embedding(cell_type.view(-1)) if self.cell_type_embedding is not None else None,
                mole=mole if mole is not None else None,
                dosage=dosage if dosage is not None else None,
                knockout=knockout if knockout is not None else None,
                single=single,
            )
        
        # h=self.block_1(cond_info_cond,
        #                timestep_embedding(timesteps, self.time_pos_dim).squeeze(),
        #                cell_type=self.cell_type_embedding(cell_type.view(-1)) if self.cell_type_embedding is not None else None)
        
        # return h.reshape(list(x_t_d.shape)+[2])
        return h