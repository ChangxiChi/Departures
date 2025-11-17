from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from tqdm import tqdm
import anndata as ad
import pandas as pd
import ot
from scipy.optimize import linear_sum_assignment
from itertools import chain



class TargetModelDataset_Molecular(Dataset):
    def __init__(self, adata_treat, adata_ctrl, cell_list, mole_embed, mole_list):
        self.adata_treat = adata_treat
        self.adata_ctrl = adata_ctrl
        self.cell_expression_treat_np = adata_treat.X.toarray()
        self.cell_list = cell_list
        self.cell_type_treat = adata_treat.obs['cell_type'].values.tolist()
        self.SMILES_treat = adata_treat.obs['SMILES'].values.tolist()
        self.mole_embed_np = mole_embed.cpu().numpy()  # numpy array or torch tensor on CPU, don't stack
        self.dosage_np = np.array(adata_treat.obs['dose_val'].values.tolist())
        self.mole_list = mole_list

        self.cell_expression_ctrl_np = self.get_ctrl_expression_matrix(self.cell_type_treat)

        self.cell_expression_treat_d_np = (self.cell_expression_treat_np > 0).astype(float)
        self.cell_expression_ctrl_d_np = (self.cell_expression_ctrl_np > 0).astype(float)

        self.mole_to_idx = {mol: i for i, mol in enumerate(self.mole_list)}
        self.cell_type_to_idx = {ctype: i for i, ctype in enumerate(self.cell_list)}

        self.cell_type_indices_np = np.array([self.cell_type_to_idx[ct] for ct in self.cell_type_treat], dtype=np.int64)

    def get_ctrl_expression_matrix(self, cell_type_treat):
        ctrl_expr_dict = {
            cell_type: self.adata_ctrl[(self.adata_ctrl.obs['cell_type'] == cell_type)].X.toarray()
            for cell_type in dict.fromkeys(cell_type_treat)
        }
        ctrl_expressions = [
            ctrl_expr_dict[cell_type][np.random.randint(0, ctrl_expr_dict[cell_type].shape[0])]
            for cell_type in cell_type_treat
        ]
        return np.stack(ctrl_expressions, axis=0)

    def __len__(self):
        return self.adata_treat.shape[0]

    def __getitem__(self, idx):
        cell_type_idx = torch.tensor(self.cell_type_indices_np[idx], dtype=torch.long)
        x0 = torch.from_numpy(self.cell_expression_ctrl_np[idx]).float()
        x1 = torch.from_numpy(self.cell_expression_treat_np[idx]).float()
        mole_name = self.SMILES_treat[idx].split("|")[0]
        mole_idx = self.mole_to_idx.get(mole_name, -1)
        if mole_idx == -1:
            mole_emb = torch.zeros_like(torch.from_numpy(self.mole_embed_np[0]))
        else:
            mole_emb = torch.from_numpy(self.mole_embed_np[mole_idx])
        dosage = torch.tensor(self.dosage_np[idx]).float()
        x0_d = torch.from_numpy(self.cell_expression_ctrl_d_np[idx]).float()
        x1_d = torch.from_numpy(self.cell_expression_treat_d_np[idx]).float()

        return {
            'cell_type': cell_type_idx,
            'x0': x0,
            'x1': x1,
            'mole': mole_emb,
            'dosage': dosage,
            'x0_d': x0_d,
            'x1_d': x1_d,
        }

    def resample(self):
        self.cell_expression_ctrl_np = self.get_ctrl_expression_matrix(self.cell_type_treat)



class TargetModelDataset_Gene(Dataset):
    def __init__(self, adata_treat, adata_ctrl, cell_list, gene_list):
        self.gene_list = gene_list
        self.adata_treat = adata_treat
        self.adata_ctrl = adata_ctrl
        self.cell_expression_treat = torch.tensor(adata_treat.X.toarray()).float()
        self.cell_list = cell_list
        self.cell_type_treat = adata_treat.obs['cell_type'].values.tolist()
        self.knockout_treat = adata_treat.obs['knockout'].values.tolist()

        self.cell_expression_ctrl = self.get_ctrl_expression_matrix(self.cell_type_treat, adata_ctrl).float()

        self.cell_expression_treat_d = (self.cell_expression_treat > 0).float()
        self.cell_expression_ctrl_d = (self.cell_expression_ctrl > 0).float()

        self.cell_type_to_idx = {ctype: i for i, ctype in enumerate(self.cell_list)}
        self.gene_to_idx = {gene: i for i, gene in enumerate(self.gene_list)}

        self.cell_type_indices = torch.tensor([
            self.cell_type_to_idx[ct] for ct in self.cell_type_treat
        ], dtype=torch.long)

        self.knockout_indices = torch.tensor([
            self.gene_to_idx[k.split("+")[0] if k.split("+")[0] != 'ctrl' else k.split("+")[1]]
            for k in self.knockout_treat
        ], dtype=torch.long)

    def get_ctrl_expression_matrix(self, cell_type_treat, adata_ctrl):
        ctrl_expr_dict = {
            cell_type: torch.tensor(adata_ctrl[(adata_ctrl.obs['cell_type'] == cell_type)].X.toarray())
            for cell_type in dict.fromkeys(cell_type_treat)
        }
        ctrl_expressions = [
            ctrl_expr_dict[cell_type][np.random.randint(0, ctrl_expr_dict[cell_type].shape[0])]
            for cell_type in cell_type_treat
        ]
        return torch.stack(ctrl_expressions, dim=0)

    def __len__(self):
        return self.adata_treat.shape[0]

    def __getitem__(self, idx):
        return {
            'cell_type': self.cell_type_indices[idx],
            'x0': self.cell_expression_ctrl[idx, :],
            'x1': self.cell_expression_treat[idx, :],
            'knockout': self.knockout_indices[idx],
            'x0_d': self.cell_expression_ctrl_d[idx, :],
            'x1_d': self.cell_expression_treat_d[idx, :],
        }
    
    def resample(self):
        self.cell_expression_ctrl_np = self.get_ctrl_expression_matrix(self.cell_type_treat, self.adata_ctrl)


def return_dataloader(adata_treat, adata_ctrl, cell_type, mole_embed=None, mole_list=None, gene_name=None, pert_type="molecular", batch_size=32):
    if pert_type == "molecular":
        dataset = TargetModelDataset_Molecular(adata_treat, adata_ctrl, cell_type, mole_embed, mole_list)
    else:
        dataset = TargetModelDataset_Gene(adata_treat, adata_ctrl, cell_type, gene_name)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, adata_treat, adata_ctrl, cell_type, mole_embed=None, mole_list=None, gene_name=None, pert_type="molecular", batch_size=32):
        super().__init__()
        self.adata_treat = adata_treat
        self.adata_ctrl = adata_ctrl
        self.cell_type = cell_type
        self.mole_embed = mole_embed
        self.mole_list = mole_list
        self.gene_name = gene_name
        self.pert_type = pert_type
        self.batch_size = batch_size

    def setup(self, stage=None):
        if self.pert_type == "molecular":
            self.train_dataset = TargetModelDataset_Molecular(self.adata_treat, self.adata_ctrl, self.cell_type, self.mole_embed, self.mole_list)
        else:
            self.train_dataset = TargetModelDataset_Gene(self.adata_treat, self.adata_ctrl, self.cell_type, self.gene_name)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)




class TargetModelDataset_Molecular_OT(Dataset):
    def __init__(self, pert_data):
        self.pert_data = pert_data
        self.cell_list = pert_data.cell_type
        self.mole_list = pert_data.mole
        self.mole_embed_np = pert_data.mole_embed.cpu().numpy()

        self.mole_to_idx = {mol: i for i, mol in enumerate(self.mole_list)}
        self.cell_type_to_idx = {ctype: i for i, ctype in enumerate(self.cell_list)}

        # self.resample_OT()

    def __len__(self):
        return self.pert_data.train_cell_treated.shape[0]

    def __getitem__(self, idx):
        cell_type_idx = torch.tensor(self.cell_type_indices_np[idx], dtype=torch.long)
        x0 = torch.from_numpy(self.cell_expression_ctrl_np[idx]).float()
        x1 = torch.from_numpy(self.cell_expression_treat_np[idx]).float()
        mole_name = self.SMILES_treat[idx].split("|")[0]
        mole_idx = self.mole_to_idx.get(mole_name, -1)
        if mole_idx == -1:
            mole_emb = torch.zeros_like(torch.from_numpy(self.mole_embed_np[0]))
        else:
            mole_emb = torch.from_numpy(self.mole_embed_np[mole_idx])
        dosage = torch.tensor(self.dosage_np[idx]).float()
        x0_d = torch.from_numpy(self.cell_expression_ctrl_d_np[idx]).float()
        x1_d = torch.from_numpy(self.cell_expression_treat_d_np[idx]).float()

        return {
            'cell_type': cell_type_idx,
            'x0': x0,
            'x1': x1,
            'mole': mole_emb,
            'dosage': dosage,
            'x0_d': x0_d,
            'x1_d': x1_d,
        }
    
    def resample_OT(self):
        treated_adata_list=[]
        control_adata_list=[]
        Cell_type_list=[]
        Dosage_list=[]
        smiles_list=[]


        test_cells=self.pert_data.train_cell_treated
        test_smiles=list(test_cells.obs['SMILES'].unique())

        # treated_adata = ad.AnnData(
        #     X=np.empty(
        #     (0, self.pert_data.adata.shape[1])),
        #     var=self.pert_data.adata.var.copy()
        # )
        
        # control_adata = ad.AnnData(
        #     X=np.empty(
        #     (0, self.pert_data.adata.shape[1])),
        #     var=self.pert_data.adata.var.copy()
        # )

        # treated_adata.obs["cell_type"] = pd.Series(dtype="object")
        # treated_adata.obs["SMILES"] = pd.Series(dtype="object")
        # treated_adata.obs["dose_val"] = pd.Series(dtype="object")


        for smiles in tqdm(test_smiles, desc="Finding OT..."):
            current_test_cell = test_cells[test_cells.obs['SMILES'] == smiles]
            current_cell_type = list(current_test_cell.obs['cell_type'].unique())

            for Cell_type in current_cell_type:
                specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == Cell_type]
                all_dosage = list(specific_type_cell.obs['dose_val'].unique())

                cells_ctrl = self.pert_data.train_cell_control[self.pert_data.train_cell_control.obs['cell_type'] == Cell_type]
                cells_ctrl_expr = cells_ctrl.X.toarray()

                for Dosage in all_dosage:
                    specific_type_dosage_cell = specific_type_cell[specific_type_cell.obs['dose_val'] == Dosage]
                    
                    # random_indices = np.random.choice(cells_ctrl_expr.shape[0], 
                    #                                     size=specific_type_dosage_cell.shape[0], 
                    #                                     replace=True)
                        
                    # gene_expr_ctrl = cells_ctrl[random_indices].X.toarray()

                    # gene_expr_ctrl = cells_ctrl_expr[random_indices]

                    Y = specific_type_dosage_cell.X.toarray()
                    # X = gene_expr_ctrl
                    X = cells_ctrl_expr

                    # epsilon = 1e-8
                    # Y = Y + epsilon
                    # X = X + epsilon

                    a = ot.unif(Y.shape[0])
                    b = ot.unif(X.shape[0])

                    M = ot.dist(Y, X, metric='euclidean')**2  # shape (m, n)
                    
                    M=M/M.max()
                    G = ot.sinkhorn(a, b, M, reg=0.1)

                    G_flat = G.flatten()
                    G_flat /= G_flat.sum()  # ensure sum to 1

                    # Sample k pairs based on G as joint distribution
                    indices = np.arange(G.size)
                    sampled_idx = np.random.choice(indices, size=specific_type_dosage_cell.shape[0], p=G_flat)

                    # Recover (i, j) indices
                    rows, cols = np.unravel_index(sampled_idx, G.shape)
                    matched_X = X[cols]
                    matched_Y = Y[rows]

                    n=matched_Y.shape[0]
                    treated_adata_list.append(matched_Y)
                    control_adata_list.append(matched_X)
                    Cell_type_list.append([Cell_type]*n)
                    Dosage_list.append([Dosage]*n)
                    smiles_list.append([smiles]*n)

                    # new_treated_adata = ad.AnnData(X=matched_Y, var=treated_adata.var.copy())
                    # new_treated_adata.obs['cell_type'] = Cell_type
                    # new_treated_adata.obs['dose_val'] = Dosage
                    # new_treated_adata.obs['SMILES'] = smiles
                    # treated_adata_list.append(new_treated_adata)
                    # # treated_adata = ad.concat([treated_adata, new_treated_adata],axis=0,join="outer")

                    # new_control_adata = ad.AnnData(X=matched_X, var=control_adata.var.copy())
                    # new_control_adata.obs['cell_type'] = Cell_type
                    # new_control_adata.obs['knockout'] = 'ctrl'
                    # control_adata_list.append(new_control_adata)
                    # # control_adata = ad.concat([control_adata, new_control_adata],axis=0,join="outer")
        
        # treated_adata = ad.concat(treated_adata_list, axis=0, join="outer")
        # control_adata = ad.concat(control_adata_list, axis=0, join="outer")

        # treated_adata.obs_names = np.arange(treated_adata.n_obs).astype(str)
        # control_adata.obs_names = np.arange(control_adata.n_obs).astype(str)

        # treated_adata.var=self.pert_data.adata.var
        # control_adata.var=self.pert_data.adata.var
        
        # self.cell_expression_treat_np = treated_adata.X
        # self.cell_type_treat = treated_adata.obs['cell_type'].values.tolist()
        # self.SMILES_treat = treated_adata.obs['SMILES'].values.tolist()
        # self.dosage_np = np.array(treated_adata.obs['dose_val'].values.tolist())
        # self.cell_expression_ctrl_np = control_adata.X

        # self.cell_expression_treat_d_np = (self.cell_expression_treat_np > 0).astype(float)
        # self.cell_expression_ctrl_d_np = (self.cell_expression_ctrl_np > 0).astype(float)

        self.cell_expression_treat_np = np.concatenate(treated_adata_list, axis=0)
        self.cell_expression_ctrl_np = np.concatenate(control_adata_list, axis=0)
        self.cell_type_treat = list(chain.from_iterable(Cell_type_list))
        self.SMILES_treat = list(chain.from_iterable(smiles_list))
        self.dosage_np = np.array(list(chain.from_iterable(Dosage_list)))

        self.cell_expression_treat_d_np = (self.cell_expression_treat_np > 0).astype(float)
        self.cell_expression_ctrl_d_np = (self.cell_expression_ctrl_np > 0).astype(float)
        
        self.cell_type_indices_np = np.array([self.cell_type_to_idx[ct] for ct in self.cell_type_treat], dtype=np.int64)




class TargetModelDataset_Gene_OT(Dataset):
    def __init__(self, pert_data):
        self.pert_data = pert_data
        self.gene_list = pert_data.gene_name
        self.cell_list = pert_data.cell_type

        # self.resample_OT()

    def __len__(self):
        return self.pert_data.train_cell_treated.shape[0]

    def __getitem__(self, idx):
        return {
            'cell_type': self.cell_type_indices[idx],
            'x0': self.cell_expression_ctrl[idx, :],
            'x1': self.cell_expression_treat[idx, :],
            'knockout': self.knockout_indices[idx],
            'x0_d': self.cell_expression_ctrl_d[idx, :],
            'x1_d': self.cell_expression_treat_d[idx, :],
        }
    
    def resample_OT(self):
        treated_adata_list=[]
        control_adata_list=[]

        control_adata = ad.AnnData(
            X=np.empty(
            (0, self.pert_data.adata.shape[1])), 
            var=self.pert_data.adata.var.copy()
        )

        treated_adata = ad.AnnData(
            X=np.empty(
            (0, self.pert_data.adata.shape[1])), 
            var=self.pert_data.adata.var.copy()
        )

        control_adata.obs["knockout"] = pd.Series(dtype="object")
        control_adata.obs["cell_type"] = pd.Series(dtype="object")
        treated_adata.obs["knockout"] = pd.Series(dtype="object")
        treated_adata.obs["cell_type"] = pd.Series(dtype="object")

        all_cond=self.pert_data.adata.obs['knockout'].unique().tolist()
        train_cond=[x for x in all_cond if x not in self.pert_data.test_cond and x!='ctrl']

        for cond in tqdm(train_cond, desc="Finding OT..."):
            current_test_cell = self.pert_data.adata[self.pert_data.adata.obs['knockout'] == cond]
            current_cell_type = list(current_test_cell.obs['cell_type'].unique())
            for Cell_type in current_cell_type:
                cells_ctrl=self.pert_data.train_cell_control[self.pert_data.train_cell_control.obs['cell_type'] == Cell_type]
                
                specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == Cell_type]

                random_indices = np.random.choice(cells_ctrl.shape[0], 
                                                    size=specific_type_cell.shape[0], 
                                                    replace=False)
                    
                gene_expr_ctrl = torch.tensor(cells_ctrl[random_indices].X.toarray())

                Y = specific_type_cell.X.toarray()
                X = gene_expr_ctrl.cpu().numpy()

                a = ot.unif(Y.shape[0])
                b = ot.unif(X.shape[0])

                M = ot.dist(Y, X, metric='euclidean')**2  # shape (m, n)
                # M = ot.dist(Y, X, metric='cosine')
                # M = ot.dist(Y, X, metric='cityblock')
                M=M/M.max()
                # G = ot.emd(a, b, M)
                G = ot.sinkhorn(a, b, M, reg=0.1)
                # pairs_idx = np.argmax(G, axis=1)
                # matched_ctrl_samples = X[pairs_idx]

                G_flat = G.flatten()
                G_flat /= G_flat.sum()  # ensure sum to 1

                # Sample k pairs based on G as joint distribution
                indices = np.arange(G.size)
                sampled_idx = np.random.choice(indices, size=specific_type_cell.shape[0], p=G_flat)

                # Recover (i, j) indices
                rows, cols = np.unravel_index(sampled_idx, G.shape)
                matched_X = X[cols]
                matched_Y = Y[rows]

                # # version2: sota, cos similarity
                # _, col_ind = linear_sum_assignment(-G)
                # matched_ctrl_samples = X[col_ind]
                
                # version3: 
                # matched_ctrl_samples = []
                # topk=5
                # for i in range(G.shape[0]):
                #     topk_idx = np.argsort(G[i])[::-1][:topk]  
                #     selected_j = np.random.choice(topk_idx)
                #     matched_ctrl_samples.append(X[selected_j])

                # matched_ctrl_samples = np.stack(matched_ctrl_samples)

                new_treated_adata = ad.AnnData(X=matched_Y, var=treated_adata.var.copy())
                new_treated_adata.obs['cell_type'] = Cell_type
                new_treated_adata.obs['knockout'] = cond
                treated_adata_list.append(new_treated_adata)
                # treated_adata = ad.concat([treated_adata, new_treated_adata],axis=0,join="outer")

                new_control_adata = ad.AnnData(X=matched_X, var=control_adata.var.copy())
                new_control_adata.obs['cell_type'] = Cell_type
                new_control_adata.obs['knockout'] = 'ctrl'
                control_adata_list.append(new_control_adata)
                # control_adata = ad.concat([control_adata, new_control_adata],axis=0,join="outer")

        treated_adata = ad.concat(treated_adata_list, axis=0, join="outer")
        control_adata = ad.concat(control_adata_list, axis=0, join="outer")

        treated_adata.obs_names = np.arange(treated_adata.n_obs).astype(str)
        control_adata.obs_names = np.arange(control_adata.n_obs).astype(str)

        treated_adata.var=self.pert_data.adata.var
        control_adata.var=self.pert_data.adata.var

        # self.adata_treat = treated_adata
        self.cell_expression_treat = torch.tensor(treated_adata.X)
        self.cell_type_treat = treated_adata.obs['cell_type'].values.tolist()
        self.knockout_treat = treated_adata.obs['knockout'].values.tolist()

        self.cell_expression_ctrl = torch.tensor(control_adata.X)

        self.cell_expression_treat_d = (self.cell_expression_treat > 0).float()
        self.cell_expression_ctrl_d = (self.cell_expression_ctrl > 0).float()

        self.cell_type_to_idx = {ctype: i for i, ctype in enumerate(self.cell_list)}
        self.gene_to_idx = {gene: i for i, gene in enumerate(self.gene_list)}

        self.cell_type_indices = torch.tensor([
            self.cell_type_to_idx[ct] for ct in self.cell_type_treat
        ], dtype=torch.long)

        self.knockout_indices = torch.tensor([
            self.gene_to_idx[k.split("+")[0] if k.split("+")[0] != 'ctrl' else k.split("+")[1]]
            for k in self.knockout_treat
        ], dtype=torch.long)


def return_dataloader_OT(adata_treat, adata_ctrl, cell_type, mole_embed=None, mole_list=None, gene_name=None, pert_type="molecular", batch_size=32):
    if pert_type == "molecular":
        dataset = TargetModelDataset_Molecular_OT(adata_treat, adata_ctrl, cell_type, mole_embed, mole_list)
    else:
        dataset = TargetModelDataset_Gene_OT(adata_treat, adata_ctrl, cell_type, gene_name)
    

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)


class MyDataModule_OT(pl.LightningDataModule):
    def __init__(self, pert_data, pert_type="molecular", batch_size=32):
        super().__init__()
        self.pert_data=pert_data
        self.pert_type=pert_type
        self.batch_size=batch_size

    def setup(self, stage=None):
        if self.pert_type == "molecular":
            self.train_dataset = TargetModelDataset_Molecular_OT(self.pert_data)
        else:
            self.train_dataset = TargetModelDataset_Gene_OT(self.pert_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
