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

void fill_edges_community_ids_cuda
(
    GraphElem2* commIdKeys, 
    GraphElem*  edges,
    GraphElem*  indices,
    GraphElem** commIdsPtr,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    const GraphElem& V0,
    const GraphElem& nv_per_device,
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
    const GraphElem& V0,
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
    const GraphElem& V0,
    cudaStream_t stream = 0
);

void compute_community_weights_cuda
(
    GraphWeight** commWeights,
    GraphElem* commIds, 
    GraphWeight* vertexWeights,
    const GraphElem& nv,
    const GraphElem& nv_per_device,
    cudaStream_t stream = 0
);

void singleton_partition_cuda
(
    GraphElem* commIds,
    GraphElem* newCommIds, 
    GraphWeight* commWeights, 
    GraphWeight* vertexWeights, 
    const GraphElem& nv,
    const GraphElem& V0,  
    cudaStream_t stream = 0
);

GraphElem max_order_cuda
(
    GraphElem* indices,
    const GraphElem& nv,
    cudaStream_t stream = 0
);
/*
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
*/
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
    const GraphElem& V0,
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
    const GraphElem& V0, 
    cudaStream_t stream = 0
);

void build_local_commid_offsets_cuda
(
    GraphElem*  localOffsets,
    GraphElem*  localCommNums,
    GraphElem*  edges,
    GraphElem*  indices,
    GraphElem** commIdsPtr,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    const GraphElem& V0,
    const GraphElem& nv_per_device,
    cudaStream_t stream = 0
);

void louvain_update_cuda
(
    GraphElem*    localCommOffsets,
    GraphElem*    localCommNums,
    GraphElem*    edges,
    GraphWeight*  edgeWeights,
    GraphElem*    indices,
    GraphWeight*  vertexWeights,
    GraphElem*    commIds,
    GraphElem**   commIdsPtr,
    GraphWeight** commWeightsPtr,
    GraphElem*    newCommIds,
    const GraphWeight& mass,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    const GraphElem& V0,
    const GraphElem& nv_per_device,
    cudaStream_t stream = 0
);
/*
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
*/
void update_community_weights_cuda
(
    GraphWeight** commWeightsPtr,
    GraphElem* commIds,
    GraphElem* newCommIds,
    GraphWeight* vertexWeights,
    const GraphElem& nv,
     const GraphElem& nv_per_device,
    cudaStream_t stream = 0
);

template<typename T>
void exchange_vector_cuda
(
    T* dest,
    T* src,
    const GraphElem& nv,
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
    GraphWeight*  mod,
    GraphElem*    edges,
    GraphWeight*  edgeWeights,
    GraphElem*    indices,
    GraphElem*    commIds,
    GraphElem**   commIdsPtr,
    GraphWeight*  commWeights,
    GraphWeight** commWeightsPtr,
    GraphElem*    localCommOffsets,
    GraphElem*    localCommNums, 
    const GraphWeight& mass,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    const GraphElem& V0,
    const GraphElem& nv_per_device,
    cudaStream_t stream = 0
);

GraphWeight compute_modularity_cuda
(
    GraphWeight* mod,
    const GraphElem& nv,
    cudaStream_t stream = 0
);

#ifdef MULTIPHASE
void fill_vertex_index_cuda
(
    GraphElem* vertex_index,
    const GraphElem& nv,
    const GraphElem& V0,
    cudaStream_t stream = 0
);
/*
GraphElem build_new_vertex_id_cuda
( 
    GraphElem* commIds,
    GraphElem* vertexOffsets,
    GraphElem* newNv, 
    GraphElem* vertexIds, 
    const GraphElem& nv,
    cudaStream_t stream = 0
);

void compress_edges_cuda
(
    GraphElem*   edges, 
    GraphWeight* edgeWeights, 
    GraphElem*   numEdges, 
    GraphElem*   indices,
    GraphElem*   commIds, 
    GraphElem*   localCommOffsets, 
    GraphElem*   localCommNums, 
    const GraphElem& v0, 
    const GraphElem& v1, 
    const GraphElem& e0, 
    const GraphElem& e1,
    cudaStream_t stream = 0
);*/
/*
void compress_edge_ranges_cuda
(
    GraphElem* indices, 
    GraphElem* vertexOffsets, 
    const GraphElem& nv,
    cudaStream_t stream = 0
);
*/
/*
void compress_edge_ranges_cuda
(
    GraphElem* indices,
    GraphElem*  buffer,
    GraphElem* vertexOffsets,
    const GraphElem& nv,
    cudaStream_t stream = 0
);

template<typename T>
void sort_vector_cuda
(
    T* dest,
    T* src,
    GraphElem* orders,
    const GraphElem& nv,
    cudaStream_t stream = 0
);*/
#endif

//functions not used in the Louvain algorithm
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
