#ifndef GRAPH_CUDA_HPP_
#define GRAPH_CUDA_HPP_
#include "types.hpp"
void reorder_weights_by_keys_cuda
( 
    GraphWeight* edgeWeights, 
    GraphElem* indexOrders, 
    GraphElem* indices, 
    const GraphElem& v0, 
    const GraphElem& v1,
    const GraphElem& e0, 
    const GraphElem& e1,
    cudaStream_t stream = 0
);

void reorder_edges_by_keys_cuda
(
    GraphElem* edges, 
    GraphElem* indexOrders, 
    GraphElem* indices, 
    const GraphElem& v0, 
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    cudaStream_t stream = 0
);

void fill_edges_community_ids_cuda
(
    GraphElem2* commIdKeys, 
    GraphElem* edges,
    GraphElem* indices,
    GraphElem* commIds,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    cudaStream_t stream = 0
);

void fill_index_orders_cuda
(
    GraphElem* indexOrders,
    GraphElem* indices,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    cudaStream_t stream = 0
);

void sum_vertex_weights_cuda
(
    GraphWeight* vertex_weights,
    GraphWeight* weights,
    GraphElem*   indices,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    cudaStream_t stream = 0
);

void compute_community_weights_cuda
(
    GraphWeight* commWeights,
    GraphElem* commIds, 
    GraphWeight* vertexWeights,
    const GraphElem& nv,
    cudaStream_t stream = 0
);

void singleton_partition_cuda
(
    GraphElem* commIds,
    GraphElem* newCommIds, 
    GraphWeight* commWeights, 
    GraphWeight* vertexWeights, 
    const GraphElem& nv, 
    cudaStream_t stream = 0
);

GraphElem max_order_cuda
(
    GraphElem* indices,
    const GraphElem& nv,
    cudaStream_t stream = 0
);

void move_index_orders_cuda
(
    GraphElem* dest,
    GraphElem* src,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    cudaStream_t stream = 0
);

void reorder_edges_by_keys_cuda
(
    GraphElem* edges, 
    GraphElem* indexOrders, 
    GraphElem* indices, 
    GraphElem* buff, 
    const GraphElem& v0, 
    const GraphElem& v1,  
    const GraphElem& e0, 
    const GraphElem& e1,
    cudaStream_t stream = 0 
);

void reorder_weights_by_keys_cuda
( 
    GraphWeight* edgeWeights, 
    GraphElem* indexOrders, 
    GraphElem* indices , 
    GraphWeight* buff, 
    const GraphElem& v0, 
    const GraphElem& v1,  
    const GraphElem& e0, 
    const GraphElem& e1,
    cudaStream_t stream = 0
);

void build_local_commid_offsets_cuda
(
    GraphElem* localOffsets,
    GraphElem* localCommNums,
    GraphElem* edges,
    GraphElem* indices,
    GraphElem* commIds,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    cudaStream_t stream = 0
);

void louvain_update_cuda
(
    GraphElem* localCommOffsets,
    GraphElem* localCommNums,
    GraphElem*   edges,
    GraphWeight* edgeWeights,
    GraphElem*   indices,
    GraphWeight* vertexWeights,
    GraphElem*   commIds,
    GraphWeight* commWeights,
    GraphElem*   newCommIds,
    const GraphWeight& mass,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    cudaStream_t stream = 0
);

void update_commids_cuda
(
    GraphElem* commIds,
    GraphElem* newCommIds,
    GraphWeight* commWeights,
    GraphWeight* vertexWeights,
    const GraphElem& v0,
    const GraphElem& v1,
    cudaStream_t stream = 0
);

GraphWeight compute_mass_cuda
(
    GraphWeight* vertexWeights,
    GraphElem nv,
    cudaStream_t stream = 0
);

void copy_vector_cuda
(
    GraphElem* dest,
    GraphElem* src,
    const GraphElem& ne_,
    cudaStream_t stream = 0
);

template<const int BlockSize, const int WarpSize>
void compute_modularity_reduce_cuda
(
    GraphWeight* mod,
    GraphElem* edges,
    GraphWeight* edgeWeights,
    GraphElem* indices,
    GraphElem* commIds,
    GraphWeight* commWeights,
    GraphElem* localCommOffsets,
    GraphElem* localCommNums, 
    const GraphWeight& mass,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    cudaStream_t stream = 0
);

GraphWeight compute_modularity_cuda
(
    GraphWeight* mod,
    const GraphElem& nv,
    cudaStream_t stream = 0
);

void scan_edge_weights_cuda
(
    GraphWeight* edgeWeights, 
    Edge* edgeList, 
    const GraphElem& e0, 
    const GraphElem& e1,
    cudaStream_t stream = 0
);

void scan_edges_cuda
(
    GraphElem* edges, 
    Edge* edgeList, 
    const GraphElem& e0, 
    const GraphElem& e1,
    cudaStream_t steam = 0
);

void max_vertex_weights_cuda
(
    GraphWeight* maxVertexWeights,
    GraphWeight* edgeWeights,
    GraphElem* indices,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    cudaStream_t stream = 0
);



#endif
