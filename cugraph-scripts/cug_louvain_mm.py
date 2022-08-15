import time
import cudf
import cugraph
import os
import sys
import scipy.io, scipy.sparse
import pandas as pd

write_comms = False
if os.path.isfile(str(sys.argv[1])) is False:
    print('File not found: ', sys.argv[1])
    sys.exit()
else:
    print('Input file found, require file in matrix-market format.')
    if len(sys.argv) > 2:
        write_comms = True
        print('To write output communities on another file.')

t0 = time.time()

# Read matrix-market using SciPy and convert to Pandas
coo_mat = scipy.io.mmread(sys.argv[1])
csr_mat = coo_mat.tocsr()
offsets = pd.Series(csr_mat.indptr)
indices = pd.Series(csr_mat.indices)
data    = pd.Series(csr_mat.data)
t3 = time.time()
print('Time taken (s) to convert from matrix-market to CSR: ', (t3-t0))


t1 = time.time()

G = cugraph.from_adjlist(offsets, indices, data) 
#G = cugraph.from_adjlist(offsets, indices, None) 
parts, modularity_score = cugraph.louvain(G, max_iter=500)

t2 = time.time()

nv = G.number_of_vertices()
print('Graph #vertices/#edges: ', nv, ',', G.number_of_edges())
print('Time taken (s) for CUDA graph preparation and Louvain: ', (t2-t1))
print('Modularity score: ', modularity_score)

# write communities to a file
if write_comms:
    t1 = time.time()
    fnm = str(sys.argv[1])+'.communities' 
    f = open(fnm, "w")
    cl = 0
    print('Writing communities line by line in a file...')
    for i in range(len(parts)):
        f.write(str(parts['partition'].iloc[i]))
        f.write("\n")
        cl = cl + 1
        if cl%10000 == 0:
            print(cl, ' lines written...')   
    f.close()
    t2 = time.time()
    print('Communities written in file: ', fnm, ", time taken (s): ", (t2-t1))
