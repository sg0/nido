#ifndef GRAPH_GPU_HPP_
#define GRAPH_GPU_HPP_
#include <cstring>
#include <vector>
#include <thrust/device_ptr.h>
#include "types.hpp"
#include "graph.hpp"
#include "cuda_wrapper.hpp"
#ifdef MULTIPHASE
#include "clustering.hpp"
#endif

template<const int NGPU>
class GraphGPU
{
  private:

    Graph* graph_;

    GraphElem nv_[NGPU], ne_[NGPU], ne_per_partition_[NGPU];

    GraphElem   *edges_[NGPU];
    GraphWeight *edgeWeights_[NGPU];

    GraphElem2 *commIdKeys_[NGPU];
    GraphElem* indexOrders_[NGPU]; 

    GraphElem* indices_[NGPU];
    GraphWeight* vertexWeights_[NGPU]; 
    GraphElem* commIds_[NGPU];
    GraphWeight* commWeights_[NGPU];
    GraphElem* newCommIds_[NGPU];
    GraphElem maxOrder_[NGPU];
    GraphWeight mass_[NGPU];

    GraphElem* indicesHost_[NGPU];
    GraphElem* edgesHost_[NGPU];
    GraphWeight* edgeWeightsHost_[NGPU];
 
    std::vector<GraphElem> vertex_partition_[NGPU];

    cudaStream_t cuStreams[4][NGPU];

    //related to sorting
    thrust::device_ptr<GraphElem> orders_ptr[NGPU]; //= thrust::device_pointer_cast(indexOrders_);
    thrust::device_ptr<GraphElem2> keys_ptr[NGPU]; // = thrust::device_pointer_cast(commIdKeys_);
    less_int2 comp;

    #ifdef MULTIPHASE
    //thrust::device_ptr<GraphElem> commIds_ptr; 
    //thrust::device_ptr<GraphElem> vertex_orders_ptr;
    void* buffer_;
    GraphElem* vertexIdsHost_[NGPU];
    GraphElem* numEdgesHost_[NGPU];
    GraphElem* sortedIndicesHost_[NGPU];
    #endif
    GraphElem determine_optimal_edges_per_batch 
    (
        const GraphElem&, 
        const GraphElem&, 
        const unsigned& size
    );

    void determine_optimal_vertex_partition
    (
        GraphElem*, 
        const GraphElem&, 
        const GraphElem&, 
        const GraphElem&, 
        std::vector<GraphElem>& partition
    );

    GraphElem max_order();
    void sum_vertex_weights();
    void compute_mass();

    #ifdef MULTIPHASE

    GraphElem sort_vertex_by_community_ids
    (
        GraphElem* vertexIds,
        GraphElem* vertexOffsets,
        GraphElem* newNv
    );
    void shuffle_edge_list
    (
        GraphElem* vertexIds,
        GraphElem* vertexOffsets,
        GraphElem* numEdges,
        GraphElem* newNv
    );
    void compress_all_edges
    (
        GraphElem* numEdges
    );

    Clustering* clusters_;

    #endif

    GraphElem e0_[NGPU], e1_[NGPU];
    GraphElem w0_[NGPU], w1_[NGPU];

    GraphElem v_base_[NGPU]; v_end_[NGPU];
    GraphElem e_base_[NGPU]; e_end_[NGPU];
  public:
    GraphGPU (Graph* graph);
    ~GraphGPU();

    void set_communtiy_ids(GraphElem* commIds);
    void singleton_partition();

    void compute_community_weights();
    void sort_edges_by_community_ids
    (
        const GraphElem& v0,   //starting global vertex index 
        const GraphElem& v1,   //ending global vertex index
        const GraphElem& e0,   //starting global edge index
        const GraphElem& e1,   //ending global edge index
        const GraphElem& e0_local  //starting local index
    );

    GraphElem* get_vertex_partition();
    GraphElem get_num_partitions();
    GraphElem get_edge_partition(const GraphElem&);

    void louvain_update
    (
        const GraphElem& v0, 
        const GraphElem& v1,
        const GraphElem& e0,
        const GraphElem& e1,
        const GraphElem& e0_local
    );

    GraphWeight compute_modularity();

    void move_edges_to_device(const GraphElem& v0, const GraphElem& v1, cudaStream_t stream = 0);
    void move_edges_to_host(const GraphElem& v0,  const GraphElem& v1, cudaStream_t stream = 0);
    void move_weights_to_device(const GraphElem& v0, const GraphElem& v1, cudaStream_t stream = 0);
    void move_weights_to_host(const GraphElem& v0, const GraphElem& v1, cudaStream_t stream = 0);

    void restore_community_ids();

    #ifdef MULTIPHASE
    bool aggregation();
    void dump_partition(const std::string&);
    #endif
};
#endif
