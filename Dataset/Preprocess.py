import sys

import torch
import numpy as np
import pandas as pd
import scanpy as sc
import json
import os
import random
import pickle
from torch.utils.data import DataLoader, Dataset
from Dataset.GRN import *
from Dataset.MoleEmb import *
from torch_geometric.utils import from_networkx

'''
data_name can be ['adamson','norman','sciplex3']

adamson:{
        type: Genetic perturbation,
        cell type num: 1
        Organism: Homo sapiens}

norman:{
        type: Genetic perturbation,
        cell type num: 1
        Organism: Homo sapiens}
        
sciplex3:{
        type: Molecular perturbation,
        cell type num: 3
        Organism: Homo sapiens # we only use human cell}
'''

class PertData:
    def __init__(self,pert_type="molecular",data_name="sciplex3",hvg_num=2000,split_ratio=0.75,threshold=0.25,threshold_co=0.45):

        current_file_path = os.path.abspath(__file__)
        self.current_dir = os.path.dirname(current_file_path)

        self.pert_type=pert_type
        self.hvg_num=hvg_num
        result_path=self.current_dir+"/result/"+data_name+"_"+str(hvg_num)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        self.testing_cond_path=result_path+"/testing_cond.pkl"  # record testing condition
        self.adata=self.get_adata(pert_type,data_name,hvg_num)


        self.add_knockout_colume()


        self.train_cond, self.train_cell, self.test_cond=self.split_dataset(pert_type=pert_type,data_name=data_name,ratio=split_ratio)
        # self.normalization_gaussian()
        # self.normalization()

        self.gene_name=self.get_gene_name(pert_type=pert_type, data_name=data_name)
        self.gene_num=hvg_num
        self.cell_type=self.get_cell_type(pert_type=pert_type, data_name=data_name)
        self.cell_type_num=len(self.cell_type)
        # Obtaining control group cells corresponding to each cell type
        # Preparing for training the source model
        # self.train_cells_control_expression,self.train_cells_control_type=self.get_control(self.cell_type)

        # excluding the control cells
        self.train_cell_control = self.maintain_ctrl()
        self.train_cell_treated = self.exclude_ctrl()


        GRN=get_GRN(adata=self.train_cell,data_name=data_name,path=self.current_dir,threshold=threshold,threshold_co=threshold_co)

        '''
        only co-expression
        '''
        # import networkx as nx
        # co_expression_edges=co_expression_GRN(adata=self.train_cell,data_name=data_name,threshold=threshold_co)
        # GRN = nx.Graph()
        # for edge in co_expression_edges:
        #     gene1, gene2, _ = edge
        #     GRN.add_edge(gene1, gene2)

        # clear attributes
        for node in GRN.nodes():
            GRN.nodes[node].clear()

        for u, v in GRN.edges():
            GRN.edges[u, v].clear()

        self.GRN = from_networkx(GRN) # graph data structure
        self.GRN.edge_index = self.GRN.edge_index.to('cuda')
        # extract molecular embedding
        if pert_type=="molecular":
            smiles=list(dict.fromkeys(self.adata.obs["SMILES"]))
            self.mole=[s.split("|")[0] for s in smiles if s!='']
            self.mole_embed=extract_mole_embed(self.mole)
            # self.mole_embed = get_rdkit_embeddings_tensor(self.mole)

            # self.test_mole=[]
            # for cond in self.test_cond:
            #     mole=self.adata.obs.loc[self.adata.obs['condition'] == cond, 'SMILES'].unique()[0]
            #     mole = mole.split("|")[0]
            #     self.test_mole.append(mole)


    # After spliting the dataset
    # excluding the contrl cells
    # for the training of target model
    def normalization_gaussian(self):
        X_min = (self.train_cell.X.toarray()).min()
        X_max = (self.train_cell.X.toarray()).max()
        self.max = np.array(X_max, dtype=np.float32)
        self.min = np.array(X_min, dtype=np.float32)

        X_norm = (self.adata.X - self.min) / (self.max - self.min)

        import scipy.sparse as sp
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()
        self.adata.X = (X_norm * 2 - 1).astype(np.float32)


    def normalization(self):
        X_max = (self.train_cell.X.toarray()).max()
        self.max = np.array(X_max, dtype=np.float32)

        X_norm = self.adata.X/ self.max

        import scipy.sparse as sp
        if sp.issparse(X_norm):
            X_norm = X_norm.toarray()
        self.adata.X = X_norm.astype(np.float32)

    def recover_from_gaussian(self):
        self.adata.X=((self.adata.X+1)/2).astype(np.float32)
        self.adata.X=self.adata.X*(self.max-self.min)+self.min

    def recover(self):
        self.adata.X=(self.adata.X*self.max).astype(np.float32)

    def maintain_ctrl(self):
        adata_filtered = self.train_cell[self.train_cell.obs['condition'] == 'ctrl']
        return adata_filtered

    def exclude_ctrl(self):
        adata_filtered = self.train_cell[self.train_cell.obs['condition'] != 'ctrl']
        return adata_filtered

    def get_gene_name(self,pert_type="molecular",data_name="sciplex3"):
        if pert_type=="gene":
            gene_name = self.adata.var['gene_name'].tolist()

        elif pert_type=="molecular":
            if data_name=="sciplex3":
                gene_name=self.adata.var.index.tolist()

        return gene_name


    def get_cell_type(self,pert_type="molecular",data_name="sciplex3"):
        if pert_type=="gene":
            try:
                cell_type=sorted(self.adata.obs['cell_type'].unique().tolist())
                return cell_type
            except KeyError:
                return ['unknow'] # ignore cell_type, consider its number as 1
        elif pert_type=="molecular":
            if data_name=="sciplex3":
                cell_type=sorted(self.adata.obs['cell_type'].unique().tolist())
            return cell_type
        else:
            return None

    def get_control(self,cell_type):
        '''
        output tensor cell_expression (N,d) and cell_type_label (N,)
        N denotes number of control cell
        '''
        print("Extracting control cells of each cell type...")
        if len(cell_type)==1:
            adata_ctrl = self.adata[self.adata.obs['condition'] == 'ctrl']
            expr_matrix = adata_ctrl.X.toarray() if hasattr(adata_ctrl.X, "toarray") else adata_ctrl.X
            # expr_tensor = torch.tensor(expr_matrix, dtype=torch.float32)
            cell_expression = torch.tensor(expr_matrix).to('cuda')
            cell_type_label = torch.full((cell_expression.shape[0],), 0).to('cuda')

        else:
            cell_expression=[]
            cell_type_label=[]
            for i in range(len(cell_type)):
                adata_ctrl=self.adata[(self.adata.obs['cell_type'] == cell_type[i]) & (self.adata.obs['condition'] == 'ctrl')]
                expr_matrix = adata_ctrl.X.toarray() if hasattr(adata_ctrl.X, "toarray") else adata_ctrl.X
                expr_matrix=torch.tensor(expr_matrix)
                cell_expression.append(expr_matrix)
                label=torch.full((expr_matrix.shape[0],), i)
                cell_type_label.append(label)

            cell_expression=torch.cat(cell_expression,dim=0).to('cuda')
            cell_type_label=torch.cat(cell_type_label,dim=0).to('cuda')
            print("Completed!")
        return cell_expression,cell_type_label


    def get_adata(self,pert_type="gene",data_name="adamson",hvg_num=2000):
        print("Loading perturbation data...")
        if pert_type=="gene":
            adata=sc.read_h5ad(self.current_dir+"/"+pert_type+"/"+data_name+".h5ad")
        elif pert_type=="molecular":
            if data_name=="sciplex3":
                adata=sc.read_h5ad(self.current_dir+"/"+pert_type+"/sciplex_complete_unlasting_ver.h5ad")

        adata.X = adata.X.toarray()
        print("Completed!")



        print("Normalizing the data and selecting highly variable genes...")
        if pert_type=="gene":
            sc.pp.log1p(adata)
            # self.adata_all = adata.copy()
            self.all_cond_raw = list(adata.obs['condition'].unique())
            if data_name == "norman":
                # only consider single genetic perturbation

                pert_gene=[]
                condition=list(dict.fromkeys(adata.obs['condition'].values))
                for cond in condition:
                    gene_name=self.extract_gene_name(cond)
                    if len(gene_name)==1:
                        pert_gene.append(gene_name[0])
                    elif len(gene_name)==2:
                        pert_gene.append(gene_name[0])
                        pert_gene.append(gene_name[1])
                    else:
                        continue

                # Remove duplicates
                pert_gene=list(dict.fromkeys(pert_gene))
                pert_gene_id = [adata.var.loc[adata.var["gene_name"] == gene].index[0] for gene in pert_gene]
                sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=None)
                remaining_hvg = adata.var.loc[~adata.var_names.isin(pert_gene_id)].copy()
                top_hvg_id = remaining_hvg.sort_values("variances", ascending=False).head(hvg_num - len(pert_gene)).index
                target=list(pert_gene_id)+list(top_hvg_id)

                adata=adata[:,target]

            elif data_name=="adamson":
                filtered_cond = list(dict.fromkeys(adata.obs['condition'].values))
                pert_gene = [self.extract_gene_name(cond)[0] for cond in filtered_cond if
                             len(self.extract_gene_name(cond)) != 0]
                # Remove duplicates
                pert_gene = list(dict.fromkeys(pert_gene))

                pert_gene_id = [adata.var.loc[adata.var["gene_name"] == gene].index[0] for gene in pert_gene]
                sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=None)
                remaining_hvg = adata.var.loc[~adata.var_names.isin(pert_gene_id)].copy()
                top_hvg_id = remaining_hvg.sort_values("variances", ascending=False).head(hvg_num - len(pert_gene)).index
                target = list(pert_gene_id) + list(top_hvg_id)

                adata = adata[:, target]
        elif pert_type=="molecular":
            if data_name == "sciplex3":
                adata.obs['condition'] = adata.obs['condition'].replace('control', 'ctrl')
                # self.adata_all = adata.copy()

                self.all_cond_raw = list(adata.obs['condition'].unique())

        print("Completed!")
        return adata


    # split dataset based on perturbations
    # return training set and testing set
    def split_dataset(self,pert_type,data_name,ratio=0.75,only_single=True):
        print("Spliting the Dataset...")
        if os.path.exists(self.testing_cond_path):
            print("Spliting result already exist!")
            with open(self.testing_cond_path, 'rb') as f:
                test_cond = pickle.load(f)

            if pert_type=="gene":
                if data_name=="adamson":
                    single_perturb=[]
                    perturbation_list = list(self.adata.obs['condition'].unique())
                    for cond in perturbation_list:
                        t = self.extract_gene_name(text=cond)
                        if len(t) <= 1:
                            single_perturb += [cond]

                    perturbation_list = list(dict.fromkeys(single_perturb))

                    perturbation_list.remove('ctrl')
                    train_cond=[item for item in perturbation_list if self.extract_gene_name(item)[0] not in test_cond]
                    train_cond.append('ctrl')

                    # split adata
                    train_cell=self.adata[self.adata.obs['condition'].isin(train_cond),:]

                    # test_cell=self.adata[self.adata.obs['condition'].isin(test_cond),:]
                    print("Completed!")
                    return train_cond, train_cell, test_cond
                else:
                    return None
            else:
                '''
                the testing conditions of SciPlex3 have already stored in adata!!!!!
                '''
                return None

        else:
            if pert_type=="gene":
                if only_single:
                    """only use single gene perturbation for training"""
                    if data_name=="adamson":
                        single_gene=[] # record single perturbation gene
                        perturbation_list=list(self.adata.obs['condition'].unique())
                        for cond in perturbation_list:
                            t = self.extract_gene_name(text=cond)
                            if len(t)==1:
                                single_gene+=t

                        single_gene=list(dict.fromkeys(single_gene))

                        # remove 'ctrl' condition
                        # 'ctrl' represent unperturbed state of cell
                        if 'ctrl' in single_gene:
                            single_gene.remove('ctrl')
                        random.shuffle(single_gene)
                        train_size=int(len(single_gene)*ratio)
                        train_cond=single_gene[:train_size]
                        test_cond=single_gene[train_size:]

                        # split adata
                        train_idx=[]
                        test_idx=[]
                        for i in range(len(list(self.adata.obs['condition']))):
                            cond = self.adata.obs['condition'][i]
                            t = self.extract_gene_name(text=cond)
                            if (len(t)==1 and (t[0] in train_cond)) or len(t)==0: #including ctrl
                                train_idx.append(i)
                            elif len(t)==1 and (t[0] in test_cond):
                                test_idx.append(i)

                        train_cell=self.adata[train_idx,:]
                        test_cell=self.adata[test_idx,:]

                        # save testing cond for prediction
                        with open(self.testing_cond_path, 'wb') as f:
                            pickle.dump(test_cond, f)
                        print("Completed!")
                        return train_cond,train_cell,test_cond
                    elif data_name=="norman":
                        train_cond=self.adata[self.adata.obs['num']!=2].obs['knockout'].unique().tolist()
                        train_cell=self.adata[self.adata.obs['num']!=2]
                        test_cond = self.adata[self.adata.obs['num'] == 2].obs['knockout'].unique().tolist()

                        print("Completed!")
                        return train_cond, train_cell, test_cond
                    else:
                        return None
                else:
                    return None

            elif pert_type=="molecular":
                if data_name=="sciplex3":
                    train_cell=self.adata[self.adata.obs['unlasting_split']=='train']
                    ood_cell=self.adata[self.adata.obs['unlasting_split'] == 'ood']
                    # save testing cond for prediction
                    smiles = list(dict.fromkeys(ood_cell.obs["SMILES"]))
                    # record OOD drugs
                    self.ood_cond = [s.split("|")[0] for s in smiles if s != '']


                    print("Completed!")
                    return None, train_cell, None
                else:
                    return None
            else:
                return None

    # waiting to be corrected
    def get_id_gene(self,id_list):
        gene_list=[]
        for id in id_list:
            gene=self.id_to_gene_map[id]
            gene_list.append(gene)

        return gene_list

    # input index
    # output gene name
    def get_idx_gene(self,idx_list):
        gene_list=[]
        for idx in idx_list:
            gene=self.gene_name[idx]
            gene_list.append(gene)

        return gene_list

    # return index in self.gene_name
    def get_gene_idx(self,gene_list):
        idx_list = []
        gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_name)}

        for gene in gene_list:
            if gene in gene_to_idx:
                idx_list.append(gene_to_idx[gene])
            else:
                print(f"{gene} can't be found!")
                sys.exit(1)

        return idx_list


    def get_averger_ctrl(self):
        ctrl = self.adata[self.adata.obs['condition'].isin(['ctrl']), :]
        control = torch.tensor(np.mean(ctrl.X.toarray(), axis=0))
        return control

    def add_knockout_colume(self):
        # adata only contain single perturbation
        knockout=[]
        num=[]
        if self.pert_type=="gene":
            all_condition=self.adata.obs['condition'].values.tolist()
            for cond in all_condition:
                knock_gene=self.extract_gene_name(cond)
                if len(knock_gene)==0:
                    knockout.append(cond)
                    num.append(0)
                elif len(knock_gene)==1:
                    knockout.append(knock_gene[0])
                    num.append(1)
                else:
                    knockout.append(cond)
                    num.append(2)
            self.adata.obs['knockout']=knockout
            self.adata.obs['num']=num

    def extract_gene_name(self,text):
        filtered_parts = [part for part in text.split('+') if part != "ctrl"]
        filtered_parts.sort()
        return filtered_parts


if __name__ == "__main__":
    pertdata=PertData(pert_type="molecular",data_name="sciplex3")
    # pertdata = PertData(pert_type="gene", data_name="adamson")