import os
import sys
import time
import cugraph
import dask_cudf
import scipy.io, scipy.sparse
import pandas as pd

from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
import cugraph.comms as Comms
import cugraph.dask as dask_cugraph

if __name__ == '__main__':
  # Convert MM to CSV
  csv_file = ""
  if (len(sys.argv) == 3):
    t0 = time.time()
    # Read matrix-market using SciPy and convert to Pandas
    coo_mat = scipy.io.mmread(sys.argv[1])
    edges = list(zip(coo_mat.row, coo_mat.col, coo_mat.data))
    df = pd.DataFrame(edges)
    csv_file = str(sys.argv[2])
    df.to_csv(csv_file, sep='\t', header=False, index=False)
    t3 = time.time()
    print('Time taken (s) to convert from matrix-market to CSV: ', (t3-t0))
  else:
    csv_file = str(sys.argv[1])
   
  cluster = LocalCUDACluster()
  client = Client(cluster)
  Comms.initialize(p2p=True) 
  t1 = time.time()

  chunksize = dask_cugraph.get_chunksize(csv_file)

  # Multi-GPU CSV reader
  e_list = dask_cudf.read_csv(csv_file, chunksize = chunksize, delimiter='\t', names=["src", "dst", "wgt"], dtype=["int64", "int64", "float64"])
  G = cugraph.DiGraph()
  G.from_dask_cudf_edgelist(e_list, source='src', destination='dst')
  
  t1 = time.time()
  
  parts, modularity_score = dask_cugraph.louvain(G, max_iter=500)
  
  t2 = time.time()
  
  nv = G.number_of_vertices()
  print('Graph #vertices/#edges: ', nv, ',', G.number_of_edges())
  print('Time taken (s) for CUDA graph preparation and Louvain: ', (t2-t1))
  print('Modularity score: ', modularity_score)
  
  Comms.destroy()
  client.close()
  cluster.close()
