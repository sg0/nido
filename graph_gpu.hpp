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
#define NGPU 8

//all indices are stored in global index
class GraphGPU
{
  private:

    Graph* graph_;

    GraphElem NV_, NE_;

    GraphElem nv_[NGPU], ne_[NGPU], ne_per_partition_[NGPU];
    GraphElem nv_per_device_;

    GraphElem   *edges_[NGPU];
    GraphWeight *edgeWeights_[NGPU];

    GraphElem2 *commIdKeys_[NGPU];
    GraphElem* indexOrders_[NGPU]; 

    GraphElem* indices_[NGPU];
    GraphWeight* vertexWeights_[NGPU]; 

    GraphElem*  commIds_[NGPU];
    GraphElem** commIdsPtr_[NGPU];

    GraphWeight*  commWeights_[NGPU];
    GraphWeight** commWeightsPtr_[NGPU];

    GraphElem* newCommIds_[NGPU];

    GraphElem* localCommNums_[NGPU];
    GraphElem* localOffsets_[NGPU];

    GraphElem maxOrder_;
    GraphWeight mass_;

    GraphElem* indicesHost_;
    GraphElem* edgesHost_;
    GraphWeight* edgeWeightsHost_;

    std::vector<GraphElem> vertex_partition_[NGPU];

    cudaStream_t cuStreams[NGPU][4];

    //related to sorting
    thrust::device_ptr<GraphElem> orders_ptr[NGPU]; //= thrust::device_pointer_cast(indexOrders_);
    thrust::device_ptr<GraphElem2> keys_ptr[NGPU]; // = thrust::device_pointer_cast(commIdKeys_);
    less_int2 comp;

    GraphElem e0_[NGPU], e1_[NGPU];  //memory position with respect to edgesHost_
    GraphElem w0_[NGPU], w1_[NGPU];  //memory position with respect to edgeWeightsHost_

    GraphElem v_base_[NGPU], v_end_[NGPU]; //first and last global indices of the vertices in a given gpu 
    GraphElem e_base_[NGPU], e_end_[NGPU]; //firt and last global indices of the edges in a give gpu

    GraphElem maxPartitions_;

    GraphElem ne_per_partition_cap_;
    #ifdef MULTIPHASE
    void*       buffer_;
    GraphElem*  commIdsHost_;
    GraphElem*  vertexIdsHost_;
    GraphElem*  vertexIdsOffsetHost_;
    GraphElem*  numEdgesHost_;
    GraphElem*  sortedIndicesHost_;
    GraphElem2* sortedVertexIdsHost_;
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
        std::vector<GraphElem>& partition,
        const GraphElem&
    );

    GraphElem max_order();
    void sum_vertex_weights(const int&);
    void compute_mass();

    #ifdef MULTIPHASE

    GraphElem sort_vertex_by_community_ids();
    void shuffle_edge_list();
    void compress_all_edges();
    void compress_edges();

    Clustering* clusters_;

    #endif

  public:
    GraphGPU (Graph* graph);
    ~GraphGPU();

    void set_communtiy_ids(GraphElem* commIds);
    void singleton_partition();

    void compute_community_weights(const int&);
    void sort_edges_by_community_ids
    (
        const GraphElem& v0,   //starting global vertex index 
        const GraphElem& v1,   //ending global vertex index
        const GraphElem& e0,   //starting global edge index
        const GraphElem& e1,   //ending global edge index
        const GraphElem& e0_local,  //starting local index
        const int& host_id
    );

    GraphElem  get_vertex_partition(const Int& i, const int& host_id);
    GraphElem* get_vertex_partition(const int& host_id);
    GraphElem get_num_partitions(const int& host_id);
    GraphElem get_num_partitions();
    GraphElem get_edge_partition(const GraphElem&);

    void louvain_update
    (
        const GraphElem& v0, 
        const GraphElem& v1,
        const GraphElem& e0,
        const GraphElem& e1,
        const GraphElem& e0_local,
        const int& host_id
    );

    GraphWeight compute_modularity();

    void move_edges_to_device(const GraphElem& v0, const GraphElem& v1, const int& host_id, cudaStream_t stream = 0);
    void move_edges_to_host(const GraphElem& v0,  const GraphElem& v1, const int& host_id, cudaStream_t stream = 0);
    void move_weights_to_device(const GraphElem& v0, const GraphElem& v1, const int& host_id, cudaStream_t stream = 0);
    void move_weights_to_host(const GraphElem& v0, const GraphElem& v1, const int& host_id, cudaStream_t stream = 0);

    void update_community_weights
    (
        const GraphElem& v0,
        const GraphElem& v1,
        const GraphElem& e0,
        const GraphElem& e1,
        const int& host_id
    );

    void update_community_ids
    (
        const GraphElem& v0,
        const GraphElem& v1,
        const GraphElem& u0,
        const GraphElem& u1,
        const int& host_id
    );

    void restore_community();

    #ifdef MULTIPHASE
    bool aggregation();
    void dump_partition(const std::string&);
    #endif
};
#endif
