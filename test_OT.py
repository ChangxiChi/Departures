from train import *
from Dataset.Preprocess import *
import os
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
from test_util import *

current_dir = os.path.dirname(os.path.abspath(__file__))
# config_path = os.path.join(current_dir, "conf", "adamson_test_OT.yaml")
config_path = os.path.join(current_dir, "conf", "sciplex3_test_OT.yaml")
config = OmegaConf.load(config_path)
print(config)

current_file_path = os.path.abspath(__file__) 
current_dir = os.path.dirname(current_file_path)

pert_data = PertData(
    hvg_num=config['gene_num'],
    pert_type=config['pert_type'],
    data_name=config['data_name'],
    threshold=config['threshold'],
    threshold_co=config['threshold_co']
)

time = config['time']
base_path = current_dir+"/result/" + config['data_name'] + "/"+time
result_path = base_path+ "/" + config['data_name'] + "_OT.json"

if os.path.exists(result_path):
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)
else:
    result = predict(pert_data=pert_data,base_path=base_path,args=config)


E_distance_all=[]
E_distance_DE20=[]
E_distance_DE40=[]

EMD=[]
EMD_DE20=[]
EMD_DE40=[]

# max=pert_data.max
# pert_data.recover()

if config['pert_type'] == 'molecular':
    if config['data_name'] == "sciplex3":
        if config['pred_ood'] == True:
            for mole in tqdm(pert_data.ood_cond, desc="OOD drugs conditions"):
                current_test_cell = pert_data.adata[pert_data.adata.obs['SMILES'] == mole]
                current_cell_type = list(current_test_cell.obs['cell_type'].unique())

                for cell_type in current_cell_type:
                    specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == cell_type]
                    all_dosage = list(specific_type_cell.obs['dose_val'].unique())
                    cell_type_idx = pert_data.cell_type.index(cell_type)
                    for dosage in all_dosage:
                        pcc_delta_value = []

                        specific_type_dosage_cell = specific_type_cell[specific_type_cell.obs['dose_val'] == dosage]
                        key = config['pert_type'] + "_" + mole + "_" + cell_type + "_" + str(dosage)

                        pred_cell_expr = result[key]*max

                        E = compute_e_distance(np.array(pred_cell_expr), specific_type_dosage_cell.X.toarray())
                        E_distance_all.append(E)

                        # DE gene
                        genes=pert_data.adata.var.index.tolist()
                        DE=pert_data.adata.uns['lincs_DEGs']
                        DE20_idx=[genes.index(g) for g in DE[cell_type+'_'+specific_type_dosage_cell.obs['condition'][0]+'_'+str(dosage)][:20]]
                        DE40_idx=[genes.index(g) for g in DE[cell_type+'_'+specific_type_dosage_cell.obs['condition'][0]+'_'+str(dosage)][:40]]

                        # calculate E-distance
                        E_DE20 = compute_e_distance(np.array(pred_cell_expr)[:, DE20_idx],
                                                    specific_type_dosage_cell.X.toarray()[:, DE20_idx])
                        E_distance_DE20.append(E_DE20)

                        E_DE40 = compute_e_distance(np.array(pred_cell_expr)[:, DE40_idx],
                                                    specific_type_dosage_cell.X.toarray()[:, DE40_idx])
                        E_distance_DE40.append(E_DE40)

                        distances = [
                            wasserstein_distance(specific_type_cell.X.toarray()[:, i], np.array(pred_cell_expr)[:, i])
                            for i in range(specific_type_cell.shape[1])
                        ]
                        EMD.append(np.mean(distances))

                        distances = [
                            wasserstein_distance(specific_type_cell.X.toarray()[:, i], np.array(pred_cell_expr)[:, i])
                            for i in DE20_idx
                        ]
                        EMD_DE20.append(np.mean(distances))

                        distances = [
                            wasserstein_distance(specific_type_cell.X.toarray()[:, i], np.array(pred_cell_expr)[:, i])
                            for i in DE40_idx
                        ]
                        EMD_DE40.append(np.mean(distances))

        else:
            '''unseen drug covariate'''
            test_cells = pert_data.adata[pert_data.adata.obs['unlasting_split'] == 'test']
            test_smiles = list(test_cells.obs['SMILES'].unique())

            for mole in tqdm(test_smiles, desc="Unseen drug covariate conditions"):
                current_test_cell = test_cells[test_cells.obs['SMILES'] == mole]
                current_cell_type = list(current_test_cell.obs['cell_type'].unique())

                for cell_type in current_cell_type:
                    specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == cell_type]

                    all_dosage = list(specific_type_cell.obs['dose_val'].unique())
                    cell_type_idx = pert_data.cell_type.index(cell_type)
                    for dosage in all_dosage:
                        specific_type_dosage_cell = specific_type_cell[specific_type_cell.obs['dose_val'] == dosage]

                        if specific_type_dosage_cell.shape[0] < 10:
                            continue

                        key = config['pert_type'] + "_" + mole + "_" + cell_type + "_" + str(dosage)

                        pred_cell_expr = result[key]
                        pred_cell_expr = np.array(pred_cell_expr)
                        # pred_cell_expr = pred_cell_expr * max
                        pred_cell_expr[pred_cell_expr < 0] = 0

                        # calculate E-distance
                        E = compute_e_distance(np.array(pred_cell_expr), specific_type_dosage_cell.X.toarray())
                        E_distance_all.append(E)

                        """
                        DE
                        """
                        genes=pert_data.adata.var.index.tolist()
                        DE=pert_data.adata.uns['lincs_DEGs']
                        DE20_idx=[genes.index(g) for g in DE[cell_type+'_'+specific_type_dosage_cell.obs['condition'][0]+'_'+str(dosage)][:20]]
                        DE40_idx=[genes.index(g) for g in DE[cell_type+'_'+specific_type_dosage_cell.obs['condition'][0]+'_'+str(dosage)][:40]]

                        # calculate E-distance
                        E_DE20 = compute_e_distance(np.array(pred_cell_expr)[:, DE20_idx],
                                                    specific_type_dosage_cell.X.toarray()[:, DE20_idx])
                        E_distance_DE20.append(E_DE20)


                        E_DE40 = compute_e_distance(np.array(pred_cell_expr)[:, DE40_idx],
                                                    specific_type_dosage_cell.X.toarray()[:, DE40_idx])
                        E_distance_DE40.append(E_DE40)

                        real_array = specific_type_cell.X.toarray()
                        pred_array = np.array(pred_cell_expr)

                        EMD.append(np.mean([
                            wasserstein_distance(real_array[:, i], pred_array[:, i])
                            for i in range(real_array.shape[1])
                        ]))

                        # DE20
                        EMD_DE20.append(np.mean([
                            wasserstein_distance(real_array[:, i], pred_array[:, i])
                            for i in DE20_idx
                        ]))

                        # DE40
                        EMD_DE40.append(np.mean([
                            wasserstein_distance(real_array[:, i], pred_array[:, i])
                            for i in DE40_idx
                        ]))

else:
    for cond in tqdm(pert_data.test_cond, desc="Perturbation conditions"):
        current_test_cell = pert_data.adata[pert_data.adata.obs['knockout'] == cond]
        current_cell_type = list(current_test_cell.obs['cell_type'].unique())
        for cell_type in current_cell_type:
            specific_type_cell = current_test_cell[current_test_cell.obs['cell_type'] == cell_type]
            
            if specific_type_cell.shape[0]<10:
                continue

            key = config['pert_type'] + "_" + cond + "_" + cell_type
            pred_cell_expr = result[key]
            pred_cell_expr = np.array(pred_cell_expr)
            # pred_cell_expr = pred_cell_expr * max
            pred_cell_expr[pred_cell_expr < 0] = 0

            control_type_cell = pert_data.adata[pert_data.adata.obs['cell_type'] == cell_type]
            control_type_cell = control_type_cell[control_type_cell.obs['condition'] == 'ctrl']

            # calculate E-distance
            E=compute_e_distance(np.array(pred_cell_expr), specific_type_cell.X.toarray())
            E_distance_all.append(E)

            """
            Calcualte Differential Expression Gene
            """
            # abs
            gene_list=pert_data.adata.var.index.tolist()
            if config['data_name']=="adamson":
                DE_gene=pert_data.adata.uns['rank_genes_groups_cov_all'][cell_type+'_'+cond+'+ctrl_1+1']
            elif config['data_name']=="norman":
                DE_gene = pert_data.adata.uns['rank_genes_groups_cov_all'][cell_type + '_' + cond + '_1+1']
            DE_gene=[de for de in DE_gene if de in gene_list]


            DE20_idx = [gene_list.index(g) for g in DE_gene[:20]]
            DE40_idx = [gene_list.index(g) for g in DE_gene[:40]]

            # calculate E-distance
            E_DE20 = compute_e_distance(np.array(pred_cell_expr)[:,DE20_idx], specific_type_cell.X.toarray()[:,DE20_idx])
            E_distance_DE20.append(E_DE20)

            # plot_split_violin_by_gene(np.array(pred_cell_expr)[:, DE20_idx],
            #                           specific_type_cell.X.toarray()[:, DE20_idx],
            #                           pert_data.adata.var.loc[DE_gene[:20],'gene_name'].tolist())
            # [0,1,3,5,10,11,13,16,17,22]

            E_DE40 = compute_e_distance(np.array(pred_cell_expr)[:,DE40_idx], specific_type_cell.X.toarray()[:,DE40_idx])
            E_distance_DE40.append(E_DE40)

            real_array = specific_type_cell.X.toarray()
            pred_array = np.array(pred_cell_expr)

            EMD.append(np.mean([
                wasserstein_distance(real_array[:, i], pred_array[:, i])
                for i in range(real_array.shape[1])
            ]))

            # DE20
            EMD_DE20.append(np.mean([
                wasserstein_distance(real_array[:, i], pred_array[:, i])
                for i in DE20_idx
            ]))

            # DE40
            EMD_DE40.append(np.mean([
                wasserstein_distance(real_array[:, i], pred_array[:, i])
                for i in DE40_idx
            ]))


print("Aver EMD of prediction under each condition: {:.4f} ± {:.4f}".format(
    np.mean(EMD), np.std(EMD)))
print("Aver E-distance of prediction under each condition: {:.4f} ± {:.4f}".format(
    np.mean(E_distance_all), np.std(E_distance_all)))

print("Aver EMD of DE20 prediction under each condition: {:.4f} ± {:.4f}".format(
    np.mean(EMD_DE20), np.std(EMD_DE20)))
print("Aver E-distance of DE20 prediction under each condition: {:.4f} ± {:.4f}".format(
    np.mean(E_distance_DE20), np.std(E_distance_DE20)))

print("Aver EMD of DE40 prediction under each condition: {:.4f} ± {:.4f}".format(
    np.mean(EMD_DE40), np.std(EMD_DE40)))
print("Aver E-distance of DE40 prediction under each condition: {:.4f} ± {:.4f}".format(
    np.mean(E_distance_DE40), np.std(E_distance_DE40)))
