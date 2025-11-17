from omegaconf import OmegaConf
from model import Model,train_dsbm,DSBMLightningModule
from Dataset.Preprocess import *
from Dataset.Datasets import *
import os
import scipy.sparse as sp
import anndata as ad
import ot

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "conf", "sciplex3.yaml")
config = OmegaConf.load(config_path)
print(config)

pert_data = PertData(
    hvg_num=config['gene_num'],
    pert_type=config['pert_type'],
    data_name=config['data_name'],
    threshold=config['threshold'],
    threshold_co=config['threshold_co']
)

control_adata = ad.AnnData(
    X=np.empty(
    (0, pert_data.adata.shape[1])), 
    var=pert_data.adata.var.copy()
)

treated_adata = ad.AnnData(
    X=np.empty(
    (0, pert_data.adata.shape[1])), 
    var=pert_data.adata.var.copy()
)

treated_adata_list=[]
control_adata_list=[]

if config['data_name'] == 'sciplex3':
    control_adata.obs["cell_type"] = pd.Series(dtype="object")
    control_adata.obs["SMILES"] = pd.Series(dtype="object")
    control_adata.obs["dose_val"] = pd.Series(dtype="object")

    treated_adata.obs["cell_type"] = pd.Series(dtype="object")
    treated_adata.obs["SMILES"] = pd.Series(dtype="object")
    treated_adata.obs["dose_val"] = pd.Series(dtype="object")

    '''training drug covariate'''
    # test_cells=pert_data.adata[pert_data.adata.obs['unlasting_split']=='train']
    test_cells=pert_data.train_cell_treated
    test_smiles=list(test_cells.obs['SMILES'].unique())

    for smiles in tqdm(test_smiles, desc="Training drug covariate conditions OT"):
        current_test_cell = test_cells[test_cells.obs['SMILES'] == smiles]
        current_cell_type = list(current_test_cell.obs['cell_type'].unique())

        mole = smiles.split("|")[0]
        mole_idx = pert_data.mole.index(mole)
        mole_embed = pert_data.mole_embed[mole_idx:mole_idx+1]
        mole_embed = torch.tensor(mole_embed).to('cuda')

        for Cell_type in current_cell_type:
            specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == Cell_type]
            all_dosage = list(specific_type_cell.obs['dose_val'].unique())
            cell_type_idx = pert_data.cell_type.index(Cell_type)

            cells_ctrl = pert_data.train_cell_control[
                pert_data.train_cell_control.obs['cell_type'] == Cell_type]


            for Dosage in all_dosage:
                specific_type_dosage_cell = specific_type_cell[specific_type_cell.obs['dose_val'] == Dosage]
                if specific_type_dosage_cell.shape[0]<20:
                    continue

                random_indices = np.random.choice(cells_ctrl.shape[0], 
                                                    size=specific_type_dosage_cell.shape[0], 
                                                    replace=False)
                    
                gene_expr_ctrl = cells_ctrl[random_indices].X.toarray()

                Y = specific_type_dosage_cell.X.toarray()
                X = gene_expr_ctrl

                a = ot.unif(Y.shape[0])
                b = ot.unif(X.shape[0])

                M = ot.dist(Y, X, metric='euclidean')**2  # shape (m, n)

                G = ot.emd(a, b, M)
                pairs_idx = np.argmax(G, axis=1)
                matched_ctrl_samples = X[pairs_idx]

                new_treated_adata = ad.AnnData(X=Y, var=treated_adata.var.copy())
                new_treated_adata.obs['cell_type'] = Cell_type
                new_treated_adata.obs['dose_val'] = Dosage
                new_treated_adata.obs['SMILES'] = smiles
                treated_adata_list.append(new_treated_adata)
                # treated_adata = ad.concat([treated_adata, new_treated_adata],axis=0,join="outer")

                new_control_adata = ad.AnnData(X=matched_ctrl_samples, var=control_adata.var.copy())
                new_control_adata.obs['cell_type'] = Cell_type
                new_control_adata.obs['knockout'] = 'ctrl'
                control_adata_list.append(new_control_adata)
                # control_adata = ad.concat([control_adata, new_control_adata],axis=0,join="outer")


else:
    control_adata.obs["knockout"] = pd.Series(dtype="object")
    control_adata.obs["cell_type"] = pd.Series(dtype="object")
    treated_adata.obs["knockout"] = pd.Series(dtype="object")
    treated_adata.obs["cell_type"] = pd.Series(dtype="object")

    all_cond=pert_data.adata.obs['knockout'].unique().tolist()
    train_cond=[x for x in all_cond if x not in pert_data.test_cond and x!='ctrl']

    for cond in tqdm(train_cond, desc="Perturbation conditions OT"):
        current_test_cell = pert_data.adata[pert_data.adata.obs['knockout'] == cond]
        current_cell_type = list(current_test_cell.obs['cell_type'].unique())
        for Cell_type in current_cell_type:
            cells_ctrl=pert_data.train_cell_control[pert_data.train_cell_control.obs['cell_type'] == Cell_type]
            
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

            G = ot.emd(a, b, M)
            pairs_idx = np.argmax(G, axis=1)
            matched_ctrl_samples = X[pairs_idx]

            new_treated_adata = ad.AnnData(X=Y, var=treated_adata.var.copy())
            new_treated_adata.obs['cell_type'] = Cell_type
            new_treated_adata.obs['knockout'] = cond
            treated_adata_list.append(new_treated_adata)
            # treated_adata = ad.concat([treated_adata, new_treated_adata],axis=0,join="outer")

            new_control_adata = ad.AnnData(X=matched_ctrl_samples, var=control_adata.var.copy())
            new_control_adata.obs['cell_type'] = Cell_type
            new_control_adata.obs['knockout'] = 'ctrl'
            control_adata_list.append(new_control_adata)
            # control_adata = ad.concat([control_adata, new_control_adata],axis=0,join="outer")

treated_adata = ad.concat(treated_adata_list, axis=0, join="outer")
control_adata = ad.concat(control_adata_list, axis=0, join="outer")

treated_adata.var=pert_data.adata.var
treated_adata.X = sp.csr_matrix(treated_adata.X)
control_adata.var=pert_data.adata.var
control_adata.X = sp.csr_matrix(control_adata.X)

store_treated_path = current_dir+ "/Dataset/gene/" + config['data_name'] + "_treated_OT.h5ad"
store_control_path = current_dir+ "/Dataset/gene/" + config['data_name'] + "_control_OT.h5ad"

treated_adata.write(store_treated_path)
control_adata.write(store_control_path)
