import torch
from train import *
import json
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
import numpy as np
from model import DSBMLightningModule,Model
import matplotlib.pyplot as plt


def predict(pert_data,base_path,args,pred_ood=False,sample_num=None,N=None):
    fwd_net_cfg = {
        "gene_num": args['gene_num'],
        "GRN": pert_data.GRN,
        "cell_type_num": pert_data.cell_type_num,
        "data_name": args['data_name'],
        "pert_type": args['pert_type'],
    }
    # model = DSBMLightningModule(model=Model(num_steps=50, sig=0.2, **fwd_net_cfg), lr=args['lr']).to('cuda')
    # model.load_from_checkpoint(base_path+"/model.ckpt")
    if args['OT']:
        model = DSBMLightningModule.load_from_checkpoint(
            base_path + "/model_OT.ckpt",
            model=Model(num_steps=50, sig=0.2, **fwd_net_cfg),
        ).to('cuda')
    else:
        model = DSBMLightningModule.load_from_checkpoint(
            base_path + "/model.ckpt",
            model=Model(num_steps=50, sig=0.2, **fwd_net_cfg),
        ).to('cuda')

    model.eval()
    prediction={}
    

    if args['data_name'] == 'sciplex3':
        if pred_ood==False:
            '''unseen drug covariate'''
            test_cells=pert_data.adata[pert_data.adata.obs['unlasting_split']=='test']
            test_smiles=list(test_cells.obs['SMILES'].unique())


            for smiles in tqdm(test_smiles, desc="Unseen drug covariate conditions"):
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

                        sample_num=specific_type_dosage_cell.shape[0]
                        random_indices = np.random.choice(cells_ctrl.shape[0], size=sample_num, replace=False)
                        gene_expr_ctrl = torch.tensor(cells_ctrl[random_indices].X.toarray()).to('cuda')
                        
                        cell_type = torch.tensor(cell_type_idx).to('cuda').repeat(sample_num, 1)
                        dosage = torch.tensor(Dosage, dtype=torch.float32).to('cuda').repeat(sample_num, 1)
                        mole = mole_embed.repeat(sample_num, 1)

                        predict_gene_expression,_,_ = model.predict(ctrl_expression=gene_expr_ctrl,
                                                            N=N, 
                                                            mole=mole, 
                                                            dosage=dosage,
                                                            cell_type=cell_type,
                                                            threshold=args.get('threshold_d', None))

                        key = args.pert_type + "_" + smiles + "_" + Cell_type + "_" + str(Dosage)
                        prediction[key] = predict_gene_expression.cpu().numpy().tolist()

        else: # only predict OOD drugs
            for cond in tqdm(pert_data.ood_cond, desc="OOD drugs conditions"):
                current_test_cell = pert_data.adata[pert_data.adata.obs['SMILES'] == cond]
                current_cell_type = list(current_test_cell.obs['cell_type'].unique())

                mole = current_test_cell.obs['SMILES'].unique()[0]
                mole = mole.split("|")[0]
                mole_idx = pert_data.mole.index(mole)
                mole_embed = pert_data.mole_embed[mole_idx]

                for Cell_type in current_cell_type:
                    specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == Cell_type]
                    all_dosage = list(specific_type_cell.obs['dose_val'].unique())
                    cell_type_idx = pert_data.cell_type.index(Cell_type)

                    cells_ctrl = pert_data.train_cell_control[
                        pert_data.train_cell_control.obs['cell_type'] == Cell_type]

                    sample_num=100
                    random_indices = np.random.choice(cells_ctrl.shape[0], size=sample_num, replace=False)
                    gene_expr_ctrl = torch.tensor(cells_ctrl[random_indices].X.toarray()).to('cuda')
                    
                    for Dosage in all_dosage:
                        specific_type_dosage_cell = specific_type_cell[specific_type_cell.obs['dose_val'] == Dosage]

                        cell_type = torch.tensor(cell_type_idx).to('cuda').repeat(sample_num, 1),
                        dosage = torch.tensor(Dosage, dtype=torch.float32).to('cuda').repeat(sample_num, 1),
                        mole = mole_embed.repeat(sample_num, 1),

                        predict_gene_expression,_,_ = model.predict(ctrl_expression=gene_expr_ctrl,
                                                            N=N, 
                                                            mole=mole, 
                                                            dosage=dosage,
                                                            cell_type=cell_type,
                                                            threshold=args.get('threshold_d', None))

                        key = args.pert_type + "_" + cond + "_" + Cell_type + "_" + str(Dosage)
                        prediction[key] = predict_gene_expression.cpu().numpy().tolist()

    else:
        for cond in tqdm(pert_data.test_cond, desc="Perturbation conditions"):
            current_test_cell = pert_data.adata[pert_data.adata.obs['knockout'] == cond]
            current_cell_type = list(current_test_cell.obs['cell_type'].unique())
            for Cell_type in current_cell_type:
                cells_ctrl=pert_data.train_cell_control[pert_data.train_cell_control.obs['cell_type'] == Cell_type]
                
                specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == Cell_type]
                cell_type_idx = pert_data.cell_type.index(Cell_type)

                knockout_gene_idx=pert_data.gene_name.index(cond)

                sample_num= specific_type_cell.shape[0]
                # sample_num=50
                random_indices = np.random.choice(cells_ctrl.shape[0], size=sample_num, replace=False)
                gene_expr_ctrl = torch.tensor(cells_ctrl[random_indices].X.toarray()).to('cuda')


                cell_type = torch.tensor(cell_type_idx).to('cuda').repeat(sample_num, 1)
                knockout = torch.tensor(knockout_gene_idx).to('cuda').repeat(sample_num, 1)

                predict_gene_expression,_,_ = model.predict(ctrl_expression=gene_expr_ctrl,
                                                            N=N, 
                                                            knockout=knockout, 
                                                            cell_type=cell_type,
                                                            threshold=args.get('threshold_d', None))

                key = args.pert_type + "_" + cond + "_" + Cell_type
                prediction[key] = predict_gene_expression.cpu().numpy().tolist()
    if args['OT']:
        store_path = base_path+ "/" + args.data_name + "_OT.json"
    else:
        store_path = base_path+ "/" + args.data_name + ".json"

    with open(store_path, 'w') as f:
        json.dump(prediction, f)

    return prediction


def predict_rectify(pert_data,base_path,args,pred_ood=False,sample_num=None,N=None):
    fwd_net_cfg = {
        "gene_num": args['gene_num'],
        "GRN": pert_data.GRN,
        "cell_type_num": pert_data.cell_type_num,
        "data_name": args['data_name'],
        "pert_type": args['pert_type'],
    }
    # model = DSBMLightningModule(model=Model(num_steps=50, sig=0.2, **fwd_net_cfg), lr=args['lr']).to('cuda')
    # model.load_from_checkpoint(base_path+"/model.ckpt")

    model = DSBMLightningModule.load_from_checkpoint(
        base_path + "/model_rectify.ckpt",
        model=Model(num_steps=50, sig=0.2, **fwd_net_cfg),
    ).to('cuda')

    model.eval()
    prediction={}
    

    if args['data_name'] == 'sciplex3':
        if pred_ood==False:
            '''unseen drug covariate'''
            test_cells=pert_data.adata[pert_data.adata.obs['unlasting_split']=='test']
            test_smiles=list(test_cells.obs['SMILES'].unique())


            for smiles in tqdm(test_smiles, desc="Unseen drug covariate conditions"):
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

                    sample_num=100
                    random_indices = np.random.choice(cells_ctrl.shape[0], size=sample_num, replace=False)
                    gene_expr_ctrl = torch.tensor(cells_ctrl[random_indices].X.toarray()).to('cuda')
                    
                    for Dosage in all_dosage:
                        specific_type_dosage_cell = specific_type_cell[specific_type_cell.obs['dose_val'] == Dosage]
                        cell_type = torch.tensor(cell_type_idx).to('cuda').repeat(sample_num, 1)
                        dosage = torch.tensor(Dosage, dtype=torch.float32).to('cuda').repeat(sample_num, 1)
                        mole = mole_embed.repeat(sample_num, 1)

                        predict_gene_expression,_,_ = model.predict(ctrl_expression=gene_expr_ctrl,
                                                            N=N, 
                                                            mole=mole, 
                                                            dosage=dosage,
                                                            cell_type=cell_type,
                                                            threshold=args.get('threshold_d', None))

                        key = args.pert_type + "_" + smiles + "_" + Cell_type + "_" + str(Dosage)
                        prediction[key] = predict_gene_expression.cpu().numpy().tolist()

        else: # only predict OOD drugs
            for cond in tqdm(pert_data.ood_cond, desc="OOD drugs conditions"):
                current_test_cell = pert_data.adata[pert_data.adata.obs['SMILES'] == cond]
                current_cell_type = list(current_test_cell.obs['cell_type'].unique())

                mole = current_test_cell.obs['SMILES'].unique()[0]
                mole = mole.split("|")[0]
                mole_idx = pert_data.mole.index(mole)
                mole_embed = pert_data.mole_embed[mole_idx]

                for Cell_type in current_cell_type:
                    specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == Cell_type]
                    all_dosage = list(specific_type_cell.obs['dose_val'].unique())
                    cell_type_idx = pert_data.cell_type.index(Cell_type)

                    cells_ctrl = pert_data.train_cell_control[
                        pert_data.train_cell_control.obs['cell_type'] == Cell_type]

                    sample_num=100
                    random_indices = np.random.choice(cells_ctrl.shape[0], size=sample_num, replace=False)
                    gene_expr_ctrl = torch.tensor(cells_ctrl[random_indices].X.toarray()).to('cuda')
                    
                    for Dosage in all_dosage:
                        specific_type_dosage_cell = specific_type_cell[specific_type_cell.obs['dose_val'] == Dosage]

                        cell_type = torch.tensor(cell_type_idx).to('cuda').repeat(sample_num, 1),
                        dosage = torch.tensor(Dosage, dtype=torch.float32).to('cuda').repeat(sample_num, 1),
                        mole = mole_embed.repeat(sample_num, 1),

                        predict_gene_expression,_,_ = model.predict(ctrl_expression=gene_expr_ctrl,
                                                            N=N, 
                                                            mole=mole, 
                                                            dosage=dosage,
                                                            cell_type=cell_type,
                                                            threshold=args.get('threshold_d', None))

                        key = args.pert_type + "_" + cond + "_" + Cell_type + "_" + str(Dosage)
                        prediction[key] = predict_gene_expression.cpu().numpy().tolist()

    else:
        for cond in tqdm(pert_data.test_cond, desc="Perturbation conditions"):
            current_test_cell = pert_data.adata[pert_data.adata.obs['knockout'] == cond]
            current_cell_type = list(current_test_cell.obs['cell_type'].unique())
            for Cell_type in current_cell_type:
                cells_ctrl=pert_data.train_cell_control[pert_data.train_cell_control.obs['cell_type'] == Cell_type]
                
                specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == Cell_type]
                cell_type_idx = pert_data.cell_type.index(Cell_type)

                knockout_gene_idx=pert_data.gene_name.index(cond)

                sample_num=specific_type_cell.shape[0]
                random_indices = np.random.choice(cells_ctrl.shape[0], size=sample_num, replace=False)
                gene_expr_ctrl = torch.tensor(cells_ctrl[random_indices].X.toarray()).to('cuda')


                cell_type = torch.tensor(cell_type_idx).to('cuda').repeat(sample_num, 1)
                knockout = torch.tensor(knockout_gene_idx).to('cuda').repeat(sample_num, 1)

                predict_gene_expression,_,_ = model.predict(ctrl_expression=gene_expr_ctrl,
                                                            N=N, 
                                                            knockout=knockout, 
                                                            cell_type=cell_type,
                                                            threshold=args.get('threshold_d', None))

                key = args.pert_type + "_" + cond + "_" + Cell_type
                prediction[key] = predict_gene_expression.cpu().numpy().tolist()

    store_path = base_path+ "/rectify_result/" + args.data_name + ".json"
    with open(store_path, 'w') as f:
        json.dump(prediction, f)

    return prediction


def gaussian_kernel(x, y, sigma=1.0):
    x = x[:, np.newaxis, :]  # (n,1,d)
    y = y[np.newaxis, :, :]  # (1,m,d)
    dist = np.sum((x - y) ** 2, axis=2)
    return np.exp(-dist / (2 * sigma ** 2))


def compute_mmd(P, Q, sigma=1.0):
    """
    Compute MMD between two arrays (n_samples, n_features)
    """
    K_PP = gaussian_kernel(P, P, sigma)
    K_QQ = gaussian_kernel(Q, Q, sigma)
    K_PQ = gaussian_kernel(P, Q, sigma)

    return K_PP.mean() + K_QQ.mean() - 2 * K_PQ.mean()

def compute_e_distance(X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)

    n, m = X.shape[0], Y.shape[0]

    cross_term = np.mean(cdist(X, Y, metric='euclidean'))
    X_term = np.sum(pdist(X, metric='euclidean')) * 2 / (n * (n - 1))
    Y_term = np.sum(pdist(Y, metric='euclidean')) * 2 / (m * (m - 1))

    e_dist = 2 * cross_term - X_term - Y_term
    return np.sqrt(max(e_dist, 0))

def get_top_logfc_idx(treatment, control, topk,pseudocount=1e-6):
    treatment_mean = np.mean(treatment, axis=0)
    control_mean = np.mean(control, axis=0)
    logfc = np.log2((treatment_mean + pseudocount) / (control_mean + pseudocount))
    top_idx = np.argsort(-logfc)[:topk]
    return top_idx

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0


def plot_fit(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)

    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)

    x_min, x_max = y_true.min(), y_true.max()
    x_center = (x_max + x_min) / 2
    x_half_range = (x_max - x_min) / 2
    plt.xlim(x_center - 3 * x_half_range, x_center + 3 * x_half_range)

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs True')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


import umap
import matplotlib.pyplot as plt

def umap_simple(X, labels=None):
    reducer = umap.UMAP(random_state=42)
    X_emb = reducer.fit_transform(X)
    plt.figure(figsize=(6, 5))
    if labels is not None:
        plt.scatter(X_emb[:, 0], X_emb[:, 1], c=labels, cmap='tab10', s=10)
    else:
        plt.scatter(X_emb[:, 0], X_emb[:, 1], s=10)
    plt.title("UMAP")
    plt.tight_layout()
    plt.show()


def umap_two_arrays(X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    combined = np.vstack([X, Y])
    labels = np.array([0] * len(X) + [1] * len(Y))  # 0 for X, 1 for Y

    reducer = umap.UMAP(random_state=42)
    emb = reducer.fit_transform(combined)

    plt.figure(figsize=(6, 5))
    plt.scatter(emb[:len(X), 0], emb[:len(X), 1], label='X', alpha=0.6, s=10)
    plt.scatter(emb[len(X):, 0], emb[len(X):, 1], label='Y', alpha=0.6, s=10)
    plt.legend()
    plt.title("UMAP of X and Y")
    plt.tight_layout()
    plt.show()


from sklearn.cluster import KMeans
import numpy as np

def select_diverse_by_kmeans(samples, n_samples=100):
    kmeans = KMeans(n_clusters=n_samples, random_state=0).fit(samples)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    selected = []
    for i in range(n_samples):
        cluster_points = np.where(labels == i)[0]
        if len(cluster_points) == 0:
            continue
        center = centers[i]
        dists = np.linalg.norm(samples[cluster_points] - center, axis=1)
        selected.append(cluster_points[np.argmin(dists)])
    return samples[selected]

def select_diverse_by_avg_dist(X, n_samples=100):
    D = cdist(X, X, metric='euclidean')
    avg_dists = D.mean(axis=1)
    selected_idx = np.argsort(avg_dists)[-n_samples:]
    return X[selected_idx]

from sklearn.metrics import pairwise_distances

def farthest_point_sampling(X, n_samples=100):
    n = X.shape[0]
    selected_indices = [np.random.randint(0, n)]
    distances = pairwise_distances(X, X[selected_indices])

    for _ in range(1, n_samples):
        min_distances = distances.min(axis=1)
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)
        dist_new = pairwise_distances(X, X[[next_idx]])
        distances = np.minimum(distances, dist_new)

    return X[selected_indices]


def generate_binary_samples_numpy(prob_list, n_samples):
    prob_array = np.array(prob_list).reshape(1, -1)
    rand_vals = np.random.rand(n_samples, len(prob_list))
    samples = (rand_vals < prob_array).astype(int)
    return samples

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# def plot_split_violin_by_gene(pred_array, real_array, gene_names, save_path="split_violin_genes.pdf", title=""):
#     data = []
#     for i, gene in enumerate(gene_names):
#         for val in real_array[:, i]:
#             data.append((gene, val, "Real", "left"))
#         for val in pred_array[:, i]:
#             data.append((gene, val, "Predicted", "right"))
#     df = pd.DataFrame(data, columns=["Gene", "gene expression", "Type", "Split"])

#     plt.figure(figsize=(max(10, len(gene_names) * 0.5), 6), dpi=150)
#     ax = sns.violinplot(
#         data=df, x="Gene", y="gene expression", hue="Split",
#         split=True,
#         inner=None,
#         palette={"left": "skyblue", "right": "lightcoral"},
#         linewidth=1.5,
#         legend=False,
#         scale="width"
#     )

#     plt.xticks(rotation=45, ha="right")
#     plt.tick_params(axis='y', labelsize=14)

#     plt.ylim(-0.5, 2.2)
#     plt.yticks([0, 1, 2])

#     plt.title(title)
#     plt.tight_layout()
#     plt.legend([], [], frameon=False)

#     plt.savefig(save_path, format="pdf")
#     plt.close()

#     print(f"Violin plot saved to {os.path.abspath(save_path)}")

# import numpy as np
# import matplotlib.pyplot as plt
# import umap

# def plot_umap_compare(pred_array, real_array, title="UMAP comparison", save_path="umap_compare.pdf", random_state=42,
#                       min_dist=0.5, spread=1.0, point_size=10):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import umap

#     combined = np.vstack([pred_array, real_array])

#     reducer = umap.UMAP(min_dist=min_dist, spread=spread, random_state=random_state)
#     embedding = reducer.fit_transform(combined)

#     pred_emb = embedding[:len(pred_array)]
#     real_emb = embedding[len(pred_array):]

#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.scatter(real_emb[:, 0], real_emb[:, 1], c='red', alpha=0.6, s=point_size)
#     ax.scatter(pred_emb[:, 0], pred_emb[:, 1], c='green', alpha=0.6, s=point_size)

#     ax.set_title(title)

#     ax.set_aspect('equal')
#     ax.axis('off')
#     for spine in ax.spines.values():
#         spine.set_visible(False)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     plt.close()
#     print(f"UMAP plot saved to {save_path}")


def plot_umap_compare(pred_array, real_array, title="UMAP comparison", save_path="umap_compare.pdf", random_state=42,
                      n_neighbors=15, min_dist=0.5, spread=1.0, point_size=10):
    import matplotlib.pyplot as plt
    import numpy as np
    import umap
    from sklearn.preprocessing import MinMaxScaler

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, spread=spread, random_state=random_state)
    reducer.fit(real_array)

    real_emb = reducer.embedding_
    pred_emb = reducer.transform(pred_array)

    # 对x,y坐标分别归一化到0-1范围
    scaler = MinMaxScaler()
    real_emb = scaler.fit_transform(real_emb)
    pred_emb = scaler.transform(pred_emb)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(real_emb[:, 0], real_emb[:, 1], c='red', alpha=0.6, s=point_size)
    ax.scatter(pred_emb[:, 0], pred_emb[:, 1], c='green', alpha=0.6, s=point_size)

    ax.set_title(title)

    ax.set_aspect('equal')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)  # 不显示刻度和标签
    # 保留边框
    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"UMAP plot saved to {save_path}")