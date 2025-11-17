import scanpy as sc
import scipy
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
# input_path = os.path.join(script_dir, "gene", "adamson.h5ad")
input_path = os.path.join(script_dir, "molecular", "sciplex_complete_unlasting_ver.h5ad")
output_path = input_path 

print("Loading:", input_path)
adata = sc.read_h5ad(input_path)
adata.X = scipy.sparse.csr_matrix(adata.X)
adata.write(output_path)
print("Saved to:", output_path)