#include <thrust/sort.h>
#include <thrust/execution_policy.h>
//#include <thrust/device_ptr.h>
#include <thrust/functional.h>

#include "graph_gpu.hpp"
#include "graph_cuda.hpp"
//include host side definition of graph
#include "graph.hpp"
#include "cuda_wrapper.hpp"

GraphGPU::GraphGPU(Graph* graph) : graph_(graph), 
nv_(0), ne_(0), ne_per_partition_(0), edges_(nullptr),
edgeWeights_(nullptr), commIdKeys_(nullptr),
indexOrders_(nullptr), indices_(nullptr), vertexWeights_(nullptr),
commIds_(nullptr), commWeights_(nullptr), newCommIds_(nullptr), 
maxOrder_(0), mass_(0), indicesHost_(nullptr), edgesHost_(nullptr),
edgeWeightsHost_(nullptr)
{
    nv_ = graph_->get_num_vertices();
    ne_ = graph_->get_num_edges();

    for(unsigned i = 0; i < 4; ++i)
        CudaCall(cudaStreamCreate(&cuStreams[i]));

    //alloc buffer
    CudaMalloc(indices_,       sizeof(GraphElem)*(nv_+1));
    CudaMalloc(vertexWeights_, sizeof(GraphWeight)*nv_);
    CudaMalloc(commIds_,       sizeof(GraphElem)*nv_);
    CudaMalloc(commWeights_,   sizeof(GraphWeight)*nv_);
    CudaMalloc(newCommIds_,    sizeof(GraphElem)*nv_);

    indicesHost_     = graph_->get_index_ranges();
    edgeWeightsHost_ = graph_->get_edge_weights(); 
    edgesHost_       = graph_->get_edges(); 

    CudaMemcpyAsyncHtoD(indices_, indicesHost_, sizeof(GraphElem)*(nv_+1), 0);
    CudaMemset(vertexWeights_, 0, sizeof(GraphWeight)*nv_);
    CudaMemset(commWeights_, 0, sizeof(GraphWeight)*nv_);

    maxOrder_ = max_order();

    unsigned unit_size = (sizeof(GraphElem) > sizeof(GraphWeight)) ? sizeof(GraphElem) : sizeof(GraphWeight); 
    ne_per_partition_ = determine_optimal_edges_per_batch (nv_, ne_, unit_size);

    determine_optimal_vertex_partition(indicesHost_, nv_, ne_, ne_per_partition_, vertex_partition_);

    CudaMalloc(edges_,       unit_size*ne_per_partition_);
    CudaMalloc(edgeWeights_, unit_size*ne_per_partition_);
    CudaMalloc(commIdKeys_,  sizeof(GraphElem2)*ne_per_partition_);
    CudaMalloc(indexOrders_, sizeof(GraphElem)*ne_per_partition_);

    CudaCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    orders_ptr = thrust::device_pointer_cast(indexOrders_);
    keys_ptr   = thrust::device_pointer_cast(commIdKeys_);
    /*#ifdef MULTIPHASE
    commIds_ptr       = thrust::device_pointer_cast(commIds);
    vertex_orders_ptr = thrust::device_pointer_cast(newCommIds);
    #endif*/
    sum_vertex_weights();
    compute_mass();
}

GraphGPU::~GraphGPU()
{
    for(unsigned i = 0; i < 4; ++i)
        CudaCall(cudaStreamDestroy(cuStreams[i]));

    CudaFree(edges_);
    CudaFree(edgeWeights_);
    CudaFree(commIdKeys_);
    CudaFree(indices_);
    CudaFree(vertexWeights_);
    CudaFree(commIds_);
    CudaFree(commWeights_);
    CudaFree(newCommIds_);
    CudaFree(indexOrders_);    
}

//TODO: in the future, this could be moved to a new class
GraphElem GraphGPU::determine_optimal_edges_per_batch
(
    const GraphElem& nv,
    const GraphElem& ne,
    const unsigned& unit_size
)
{
    float free_m;//,total_m,used_m;
    size_t free_t,total_t;

    CudaCall(cudaMemGetInfo(&free_t,&total_t));

    float occ_m = (uint64_t)2*(1.5*sizeof(GraphElem)+sizeof(GraphWeight))*nv/1048576.0;
    free_m =(uint64_t)free_t/1048576.0 - occ_m;

    GraphElem ne_per_partition = (GraphElem)(free_m / unit_size / 8. * 1048576.0); //7 is the minimum, i chose 8
    return ((ne_per_partition > ne) ? ne : ne_per_partition);

}

void GraphGPU::determine_optimal_vertex_partition
(
    GraphElem* indices,
    const GraphElem& nv,
    const GraphElem& ne,
    const GraphElem& ne_per_partition,
    std::vector<GraphElem>& vertex_partition
)
{
    vertex_partition.push_back(0);
    GraphElem start = 0;
    GraphElem end = 0;
    for(GraphElem idx = 1; idx <= nv; ++idx)
    {
        end = indices[idx];
        if(end - start > ne_per_partition)
        {
            vertex_partition.push_back(idx-1);
            start = indices[idx-1];
            idx--;
        }
    }
    vertex_partition.push_back(nv);
}

GraphElem* GraphGPU::get_vertex_partition()
{
    return vertex_partition_.data();
}

GraphElem GraphGPU::get_num_partitions()
{
    return vertex_partition_.size()-1;
}

GraphElem GraphGPU::get_edge_partition(const GraphElem& i)
{
    return indicesHost_[i];
}

void GraphGPU::set_communtiy_ids(GraphElem* commIds)
{
    CudaMemcpyAsyncHtoD(commIds_, commIds, sizeof(GraphElem)*nv_, 0);
    compute_community_weights();
    CudaMemcpyAsyncHtoD(newCommIds_, commIds, sizeof(GraphElem)*nv_, 0);
}

void GraphGPU::compute_community_weights()
{
    compute_community_weights_cuda(commWeights_, commIds_, vertexWeights_, nv_);
}

void GraphGPU::sort_edges_by_community_ids
(
    const GraphElem& v0,   //starting global vertex index 
    const GraphElem& v1,   //ending global vertex index
    const GraphElem& e0,   //starting global edge index
    const GraphElem& e1,   //ending global edge index
    const GraphElem& e0_local  //starting local index
)
{
    //thrust::device_ptr<GraphElem> orders_ptr = thrust::device_pointer_cast(indexOrders_);
    //thrust::device_ptr<GraphElem2> keys_ptr  = thrust::device_pointer_cast(commIdKeys_);
    //less_int2 comp;

    GraphElem ne = e1-e0;

    fill_index_orders_cuda(indexOrders_, indices_, v0, v1, e0, e1, cuStreams[0]);

    fill_edges_community_ids_cuda(commIdKeys_, edges_+e0_local, indices_, commIds_, v0, v1, e0, e1, cuStreams[1]);

    //CudaDeviceSynchronize();
    
    thrust::sort_by_key(keys_ptr, keys_ptr+ne, orders_ptr, comp);

    reorder_edges_by_keys_cuda(edges_+e0_local, indexOrders_, indices_, (GraphElem*)commIdKeys_, v0, v1,  e0, e1, cuStreams[0]);

    reorder_weights_by_keys_cuda(edgeWeights_+e0_local, indexOrders_, indices_, 
    (GraphWeight*)(((GraphElem*)commIdKeys_)+ne), v0, v1,  e0, e1, cuStreams[1]);

    build_local_commid_offsets_cuda(((GraphElem*)commIdKeys_), ((GraphElem*)commIdKeys_)+ne, edges_+e0_local, indices_, commIds_, v0, v1, e0, e1);
}

void GraphGPU::singleton_partition()
{
    singleton_partition_cuda(commIds_, newCommIds_, commWeights_, vertexWeights_, nv_);
    //copy_vector_cuda(newCommIds_, commIds_, nv_);
}

GraphElem GraphGPU::max_order()
{
    return max_order_cuda(indices_, nv_);
}

void GraphGPU::sum_vertex_weights()
{
    for(GraphElem b = 0; b < vertex_partition_.size()-1; ++b)
    {
        GraphElem v0 = vertex_partition_[b];
        GraphElem v1 = vertex_partition_[b+1];


        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];

        CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_+e0, sizeof(GraphWeight)*(e1-e0), 0);
        sum_vertex_weights_cuda(vertexWeights_, (GraphWeight*)edgeWeights_, indices_, v0, v1, e0, e1);
    }
}

void GraphGPU::louvain_update
(
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    const GraphElem& e0_local
)
{
    GraphElem ne = e1-e0;
    louvain_update_cuda(((GraphElem*)commIdKeys_), ((GraphElem*)commIdKeys_)+ne, edges_+e0_local, edgeWeights_+e0_local,
                        indices_, vertexWeights_, commIds_, commWeights_, newCommIds_, mass_, v0, v1, e0, e1);
    update_commids_cuda(commIds_, newCommIds_, commWeights_, vertexWeights_, v0, v1);
}    


void GraphGPU::compute_mass()
{
    mass_ = compute_mass_cuda(vertexWeights_, nv_);
}

GraphWeight GraphGPU::compute_modularity()
{
    GraphWeight* mod;
    CudaMalloc(mod,    sizeof(GraphWeight)*MAX_GRIDDIM*BLOCKDIM02/WARPSIZE);
    CudaMemset(mod, 0, sizeof(GraphWeight)*MAX_GRIDDIM*BLOCKDIM02/WARPSIZE);

    for(GraphElem b = 0; b < vertex_partition_.size()-1; ++b)
    {
        GraphElem v0 = vertex_partition_[b];
        GraphElem v1 = vertex_partition_[b+1];

        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];
        GraphElem ne = e1-e0;

        CudaMemcpyAsyncHtoD(edges_, edgesHost_+e0, sizeof(GraphElem)*ne, cuStreams[1]);

        CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_+e0, sizeof(GraphWeight)*ne, cuStreams[3]);

        sort_edges_by_community_ids(v0, v1, e0, e1, 0);
        
        compute_modularity_reduce_cuda<BLOCKDIM02, WARPSIZE>(mod, edges_, edgeWeights_, indices_, commIds_, commWeights_, 
        ((GraphElem*)commIdKeys_), ((GraphElem*)commIdKeys_)+ne, mass_, v0, v1, e0, e1);
    }

    GraphWeight q = compute_modularity_cuda(mod, MAX_GRIDDIM*BLOCKDIM02/WARPSIZE);

    CudaFree(mod);

    return q;
}

void GraphGPU::move_edges_to_device
(
    const GraphElem& e0,
    const GraphElem& e1, 
    cudaStream_t stream
)
{
    GraphElem ne = e1-e0;
    CudaMemcpyAsyncHtoD(edges_, edgesHost_+e0, sizeof(GraphElem)*ne, stream);
}

void GraphGPU::move_edges_to_host
(
    const GraphElem& e0,  
    const GraphElem& e1, 
    cudaStream_t stream
)
{
    GraphElem ne = e1-e0;
    CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_, sizeof(GraphElem)*ne, stream);
}

void GraphGPU::move_weights_to_device
(
    const GraphElem& e0, 
    const GraphElem& e1, 
    cudaStream_t stream
)
{
    GraphElem ne = e1-e0;
    CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_+e0, sizeof(GraphWeight)*ne, stream);
}

void GraphGPU::move_weights_to_host
(
    const GraphElem& e0, 
    const GraphElem& e1, 
    cudaStream_t stream
)
{
    GraphElem ne = e1-e0;
    CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_, sizeof(GraphWeight)*ne, stream);
}


#if 0
#ifdef MULTIPHASE
void GraphGPU::aggregation()
{
    CudaMemcpyAsyncDtoH(commIdsHost_, commIds_, sizeof(GraphElem)*nv_, cuStreams[0]);
    fill_vertex_index_cuda(newCommIds_ nv_, cuStreams[1]);
    thrust::sort_by_key(commIds_ptr, commIds_ptr+nv_, vertex_orders_ptr, thrust::less<GraphElem>()); 
    //CudaMemcpyAsyncDtoH(vertexOrders);
    build_new_vertex_id_cuda();       
}
#endif
#endif
