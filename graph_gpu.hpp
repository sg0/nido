#ifndef GRAPH_GPU_HPP_
#define GRAPH_GPU_HPP_
#include <vector>
#include <thrust/device_ptr.h>
#include "types.hpp"
#include "graph.hpp"
#include "cuda_wrapper.hpp"
#ifdef MULTIPHASE
#include "clustering.hpp"
#endif
class GraphGPU
{
  private:

    Graph* graph_;

    GraphElem nv_, ne_, ne_per_partition_;

    GraphElem   *edges_;
    GraphWeight *edgeWeights_;

    GraphElem2 *commIdKeys_;
    GraphElem* indexOrders_; 

    GraphElem* indices_;
    GraphWeight* vertexWeights_; 
    GraphElem* commIds_;
    GraphWeight* commWeights_;
    GraphElem* newCommIds_;
    GraphElem maxOrder_;
    GraphWeight mass_;

    GraphElem* indicesHost_;
    GraphElem* edgesHost_;
    GraphWeight* edgeWeightsHost_;
 
    std::vector<GraphElem> vertex_partition_;

    cudaStream_t cuStreams[4];

    //related to sorting
    thrust::device_ptr<GraphElem> orders_ptr; //= thrust::device_pointer_cast(indexOrders_);
    thrust::device_ptr<GraphElem2> keys_ptr; // = thrust::device_pointer_cast(commIdKeys_);
    less_int2 comp;

    #ifdef MULTIPHASE
    //thrust::device_ptr<GraphElem> commIds_ptr; 
    //thrust::device_ptr<GraphElem> vertex_orders_ptr;
    void* buffer_;
    GraphElem* vertexIdsHost_;
    GraphElem* numEdgesHost_;
    GraphElem* sortedIndicesHost_;
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
    #endif
};
#endif
