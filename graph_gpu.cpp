#include <thrust/sort.h>
#include <thrust/execution_policy.h>
//#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <omp.h>

#include "graph_gpu.hpp"
#include "graph_cuda.hpp"
//include host side definition of graph
#include "graph.hpp"
#include "cuda_wrapper.hpp"

//#ifdef CHECK
//#include "graph_cpu.hpp"
//#endif

GraphGPU::GraphGPU(Graph* graph) : graph_(graph), 
NV_(0), NE_(0), nv_per_device_(0), maxOrder_(0), 
mass_(0), maxPartitions_(0)
{
    NV_ = graph_->get_num_vertices();
    NE_ = graph_->get_num_edges();

    nv_per_device_ = (NV_+NGPU-1)/NGPU;

    if(nv_per_device_*(NGPU-1) >= NV_)
    {
        std::cout << "Too many GPUs are used. Some GPUs will idle\n";
        exit(-1);
    }        
    indicesHost_     = graph_->get_index_ranges();
    edgeWeightsHost_ = graph_->get_edge_weights();
    edgesHost_       = graph_->get_edges();

    #pragma omp parallel
    {
        int id =  omp_get_thread_num() % NGPU;   
        CudaCall(cudaSetDevice(id));

        for(unsigned i = 0; i < 4; ++i)
            CudaCall(cudaStreamCreate(&cuStreams[id][i]));

        e0_[id] = 0; e1_[id] = 0;
        w0_[id] = 0; w1_[id] = 0;

        v_base_[id]  = nv_per_device_*id;
        if(v_base_[id] > NV_) v_base_[id] = NV_;

        v_end_ [id]  = v_base_[id]+nv_per_device_;
        if(v_end_[id] > NV_) v_end_[id] = NV_;

        e_base_[id] = indicesHost_[v_base_[id]];
        e_end_[id]  = indicesHost_[v_end_[id]];

        nv_[id] = v_end_[id]-v_base_[id];
        ne_[id] = e_end_[id]-e_base_[id];

        GraphElem nv = nv_[id];
        GraphElem ne = ne_[id];

        //alloc buffer
        CudaMalloc(indices_[id],       sizeof(GraphElem)   *(nv+1));
        CudaMalloc(vertexWeights_[id], sizeof(GraphWeight) *nv);
        CudaMalloc(commIds_[id],       sizeof(GraphElem)   *nv);
        CudaMalloc(commWeights_[id],   sizeof(GraphWeight) *nv);
        CudaMalloc(newCommIds_[id],    sizeof(GraphElem)   *nv);
        CudaMalloc(commIdsPtr_[id],    sizeof(GraphElem*)  *NGPU);
        CudaMalloc(commWeightsPtr_[id],sizeof(GraphWeight*)*NGPU);

        CudaMemcpyAsyncHtoD(indices_[id], indicesHost_+v_base_[id], sizeof(GraphElem)*(nv+1), 0);
        CudaMemset(vertexWeights_[id], 0, sizeof(GraphWeight)*nv);
        CudaMemset(commWeights_[id], 0, sizeof(GraphWeight)*nv);

        unsigned unit_size = (sizeof(GraphElem) > sizeof(GraphWeight)) ? sizeof(GraphElem) : sizeof(GraphWeight); 
        ne_per_partition_[id] = determine_optimal_edges_per_batch (nv, ne, unit_size);
        GraphElem ne_per_partition = ne_per_partition_[id];

        determine_optimal_vertex_partition(indicesHost_, nv, ne, ne_per_partition, vertex_partition_[id], v_base_[id]);

        CudaMalloc(edges_[id],       unit_size*ne_per_partition);
        CudaMalloc(edgeWeights_[id], unit_size*ne_per_partition);
        CudaMalloc(commIdKeys_[id],  sizeof(GraphElem2)*ne_per_partition);
        CudaMalloc(indexOrders_[id], sizeof(GraphElem)*ne_per_partition);

        CudaCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

        orders_ptr[id] = thrust::device_pointer_cast(indexOrders_[id]);
        keys_ptr[id]   = thrust::device_pointer_cast(commIdKeys_[id]);

        sum_vertex_weights(id);
    }
    maxOrder_ = max_order();
    std::cout << "max order is " << maxOrder_ << std::endl;
    compute_mass();

    #ifdef MULTIPHASE
    buffer_ = malloc(unit_size*NE_);
    commIdsHost_ = new GraphElem [NV_];
    vertexIdsHost_ = new GraphElem [NV_];
    vertexIdsOffsetHost_ = new GraphElem [NV_];
    sortedVertexIdsHost_ = (GraphElem*)malloc(sizeof(GraphElem2)*(NV_+1));
    
    //CudaMallocHost(vertexIdsHost_, sizeof(GraphElem)*NV_);
    //CudaMallocHost(numEdgesHost_,  sizeof(GraphElem)*NV_);
    //CudaMallocHost(sortedIndicesHost_, sizeof(GraphElem)*(NV_+1));

    clusters_ = new Clustering(NV_);
    #endif

    for(int i = 0; i < NGPU; ++i)
    {
        CudaSetDevice(i);
        for(int j = 0; j < NGPU; ++j)
        {
            if(i != j)
                CudaCall(cudaDeviceEnablePeerAccess(j, 0));
        }
        CudaMemcpyHtoD(commIdsPtr_[i],     commIds_,     sizeof(GraphElem*)*NGPU);
        CudaMemcpyHtoD(commWeightsPtr_[i], commWeights_, sizeof(GraphWeight*)*NGPU);
    }

    maxPartitions_ = vertex_partition_[0].size();
    for(int i = 1; i < NGPU; ++i)
    {
        if(vertex_partition_[i].size() > maxPartitions_)
            maxPartitions_ = vertex_partition_[i].size();
    }
}

GraphGPU::~GraphGPU()
{
    for(int g = 0; g < NGPU; ++g)
    {
        CudaSetDevice(g);
        for(unsigned i = 0; i < 4; ++i)
            CudaCall(cudaStreamDestroy(cuStreams[g][i]));
    
        CudaFree(edges_[g]);
        CudaFree(edgeWeights_[g]);
        CudaFree(commIdKeys_[g]);
        CudaFree(indices_[g]);
        CudaFree(vertexWeights_[g]);
        CudaFree(commIds_[g]);
        CudaFree(commIdsPtr_[g]);
        CudaFree(commWeights_[g]);
        CudaFree(commWeightsPtr_[g]);
        CudaFree(newCommIds_[g]);
        CudaFree(indexOrders_[g]);
    }
    #ifdef MULTIPHASE
    free(buffer_);
    buffer_ = nullptr;
    delete [] commIdsHost_;
    delete [] vertexIdsHost_;
    delete [] vertexIdsOffsetHost_;
    free(sortedVertexIdsHost_);
    #endif    
}

//TODO: in the future, this could be moved to a new class
GraphElem GraphGPU::determine_optimal_edges_per_batch
(
    const GraphElem& nv,
    const GraphElem& ne,
    const unsigned& unit_size
)
{
    if(nv > 0)
    {
        float free_m;//,total_m,used_m;
        size_t free_t,total_t;

        CudaCall(cudaMemGetInfo(&free_t,&total_t));

        float occ_m = (uint64_t)((3*sizeof(GraphElem)+2*sizeof(GraphWeight))*nv)/1048576.0;
        free_m =(uint64_t)free_t/1048576.0 - occ_m;

        GraphElem ne_per_partition = (GraphElem)(free_m / unit_size / 8. * 1048576.0); //7 is the minimum, i chose 8
        return ((ne_per_partition > ne) ? ne : ne_per_partition);
    }
    return 0;
}

void GraphGPU::determine_optimal_vertex_partition
(
    GraphElem* indices,
    const GraphElem& nv,
    const GraphElem& ne,
    const GraphElem& ne_per_partition,
    std::vector<GraphElem>& vertex_partition,
    const GraphElem& V0
)
{
    vertex_partition.push_back(V0);
    GraphElem start = 0;
    GraphElem end = 0;
    for(GraphElem idx = 1; idx <= nv; ++idx)
    {
        end = indices[idx+V0];
        if(end - start > ne_per_partition)
        {
            vertex_partition.push_back(V0+idx-1);
            start = indices[V0+idx-1];
            idx--;
        }
    }
    vertex_partition.push_back(V0+nv);
}

GraphElem  GraphGPU::get_vertex_partition(const Int& i, const int& host_id)
{
    Int size = vertex_partition_[host_id].size();
    if(size <= i)
        return vertex_partition_[host_id][size-1];
    return vertex_partition_[host_id][i];
}

GraphElem* GraphGPU::get_vertex_partition(const int& i)
{
    return vertex_partition_[i].data();
}

GraphElem GraphGPU::get_num_partitions(const int& i)
{
    return vertex_partition_[i].size()-1;
}

GraphElem GraphGPU::get_num_partitions()
{
    return maxPartitions_-1;
}

GraphElem GraphGPU::get_edge_partition(const GraphElem& i)
{
    return indicesHost_[i];
}

void GraphGPU::set_communtiy_ids(GraphElem* commIds)
{
    for(int i = 0; i < NGPU; ++i) 
    {
        if(nv_[i] > 0)
        {
            CudaSetDevice(i);
            CudaMemcpyAsyncHtoD(commIds_[i], commIds+v_base_[i], sizeof(GraphElem)*nv_[i], 0);
            compute_community_weights(i);
            CudaMemcpyAsyncHtoD(newCommIds_[i], commIds+v_base_[i], sizeof(GraphElem)*nv_[i], 0);
            CudaDeviceSynchronize();
        }
    }
}

void GraphGPU::compute_community_weights(const int& host_id)
{
    if(nv_[host_id] > 0)
        compute_community_weights_cuda(commWeightsPtr_[host_id], commIds_[host_id], vertexWeights_[host_id], nv_[host_id], nv_per_device_);
}

void GraphGPU::sort_edges_by_community_ids
(
    const GraphElem& v0,   //starting global vertex index 
    const GraphElem& v1,   //ending global vertex index
    const GraphElem& e0,   //starting global edge index
    const GraphElem& e1,   //ending global edge index
    const GraphElem& e0_local,  //starting local index
    const int& host_id
)
{
    if(v1 > v0)
    {
        GraphElem ne = e1-e0;
        GraphElem V0 = v_base_[host_id];

        fill_index_orders_cuda(indexOrders_[host_id], indices_[host_id], v0, v1, e0, e1, V0, cuStreams[host_id][0]);

        fill_edges_community_ids_cuda(commIdKeys_[host_id], edges_[host_id]+e0_local, indices_[host_id], commIdsPtr_[host_id]
        , v0, v1, e0, e1, V0, nv_per_device_, cuStreams[host_id][1]);
    
        thrust::stable_sort_by_key(keys_ptr[host_id], keys_ptr[host_id]+ne, orders_ptr[host_id], comp);

        reorder_edges_by_keys_cuda(edges_[host_id]+e0_local, indexOrders_[host_id], indices_[host_id], 
        (GraphElem*)commIdKeys_[host_id], v0, v1,  e0, e1, V0, cuStreams[host_id][0]);

        reorder_weights_by_keys_cuda(edgeWeights_[host_id]+e0_local, indexOrders_[host_id], indices_[host_id], 
        (GraphWeight*)(((GraphElem*)commIdKeys_[host_id])+ne), v0, v1,  e0, e1, V0, cuStreams[host_id][1]);

        build_local_commid_offsets_cuda(((GraphElem*)commIdKeys_[host_id]), ((GraphElem*)commIdKeys_[host_id])+ne, 
        edges_[host_id]+e0_local, indices_[host_id], commIdsPtr_[host_id], v0, v1, e0, e1, V0, nv_per_device_);
    }
}

void GraphGPU::singleton_partition()
{
    #pragma omp parallel
    {
        int i = omp_get_thread_num() % NGPU;
        GraphElem V0 = v_base_[i];
        CudaSetDevice(i);
        if(nv_[i] > 0)
            singleton_partition_cuda(commIds_[i], newCommIds_[i], commWeights_[i], vertexWeights_[i], nv_[i], V0);

        CudaDeviceSynchronize();
    }
}

GraphElem GraphGPU::max_order()
{
    GraphElem maxs[NGPU];
    #pragma omp parallel
    {
        int i = omp_get_thread_num() % NGPU;
        CudaSetDevice(i);
        if(nv_[i] > 0)
            maxs[i] = max_order_cuda(indices_[i], nv_[i]);
        else
            maxs[i] = 0;
    }
    GraphElem max = maxs[0];
    #pragma unroll
    for(int i = 1; i < NGPU; ++i)
        if(max < maxs[i])
            max = maxs[i];
    return max;
}

void GraphGPU::sum_vertex_weights(const int& host_id)
{
    GraphElem V0 = v_base_[host_id];
    for(GraphElem b = 0; b < vertex_partition_[host_id].size()-1; ++b)
    {
        GraphElem v0 = vertex_partition_[host_id][b];
        GraphElem v1 = vertex_partition_[host_id][b+1];


        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];

        //CudaMemcpyAsyncHtoD(edgeWeights_, edgeWeightsHost_+e0, sizeof(GraphWeight)*(e1-e0), 0);
        if(v1 > v0)
        {
            move_weights_to_device(e0, e1, host_id);
            sum_vertex_weights_cuda(vertexWeights_[host_id], edgeWeights_[host_id], indices_[host_id], v0, v1, e0, e1, V0);
        }
    }
}

void GraphGPU::louvain_update
(
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    const GraphElem& e0_local,
    const int& host_id
)
{
    if(v1 > v0)
    {
        GraphElem ne = e1-e0;
        GraphElem V0 = v_base_[host_id];
        louvain_update_cuda(((GraphElem*)commIdKeys_[host_id]), ((GraphElem*)commIdKeys_[host_id])+ne, 
                              edges_[host_id]+e0_local, edgeWeights_[host_id]+e0_local, indices_[host_id], 
                              vertexWeights_[host_id], commIds_[host_id], commIdsPtr_[host_id], commWeightsPtr_[host_id], 
                              newCommIds_[host_id], mass_, v0, v1, e0, e1, V0, nv_per_device_);
    }
}    

void GraphGPU::update_community_weights
(
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    const int& host_id
)
{
    GraphElem nv = v1-v0;
    GraphElem V0 = v_base_[host_id];
    if(nv > 0)
        update_community_weights_cuda(commWeightsPtr_[host_id], commIds_[host_id]+v0-V0, newCommIds_[host_id]+v0-V0, 
                                      vertexWeights_[host_id]+v0-V0, nv, nv_per_device_);
 
}

void GraphGPU::update_community_ids
(
    const GraphElem& v0, 
    const GraphElem& v1,
    const GraphElem& u0,
    const GraphElem& u1,
    const int& host_id
)
{
    GraphElem nv = v1-v0;
    GraphElem V0 = v_base_[host_id];
    if(nv > 0)
        exchange_vector_cuda<GraphElem>(commIds_[host_id]+v0-V0, newCommIds_[host_id]+v0-V0, nv);
}

void GraphGPU::restore_community() 
{
    for(int i = 0; i < NGPU; ++i)
    {
        if(nv_[i] > 0)
        {
            CudaSetDevice(i);
            update_community_weights_cuda(commWeightsPtr_[i], commIds_[i], newCommIds_[i], vertexWeights_[i], nv_[i], nv_per_device_);
            CudaDeviceSynchronize();
        }
    }

    #pragma omp parallel
    {
        int i = omp_get_thread_num() % NGPU;
        CudaSetDevice(i);
        if(nv_[i] > 0)
            exchange_vector_cuda<GraphElem>(commIds_[i], newCommIds_[i], nv_[i]);
        CudaDeviceSynchronize();
    }
}

void GraphGPU::compute_mass()
{
    mass_ = 0;
    #pragma omp parallel
    {
        int i = omp_get_thread_num() % NGPU;
        CudaSetDevice(i);
        if(nv_[i] > 0)
        {
            GraphWeight m = compute_mass_cuda(vertexWeights_[i], nv_[i]);
            #pragma omp critical
            mass_ += m;
        }
    }
    std::cout << mass_ << std::endl;
}

GraphWeight GraphGPU::compute_modularity()
{
    GraphWeight q = 0.;
    #pragma omp parallel
    {
        int id = omp_get_thread_num() % NGPU;
        CudaSetDevice(id);
 
        GraphWeight* mod;
        GraphElem num = MAX_GRIDDIM*BLOCKDIM02/WARPSIZE;

        CudaMalloc(mod,    sizeof(GraphWeight)*num);
        CudaMemset(mod, 0, sizeof(GraphWeight)*num);
        GraphElem V0 = v_base_[id];

        for(GraphElem b = 0; b < vertex_partition_[id].size()-1; ++b)
        {
            GraphElem v0 = vertex_partition_[id][b];
            GraphElem v1 = vertex_partition_[id][b+1];

            GraphElem e0 = indicesHost_[v0];
            GraphElem e1 = indicesHost_[v1];
            GraphElem ne = e1-e0;

            move_edges_to_device(e0, e1, id, cuStreams[id][1]);

            move_weights_to_device(e0, e1, id, cuStreams[id][2]);

            sort_edges_by_community_ids(v0, v1, e0, e1, 0, id);
            if(v1 > v0)      
                compute_modularity_reduce_cuda<BLOCKDIM02, WARPSIZE>(mod, edges_[id], edgeWeights_[id], indices_[id], 
                commIds_[id], commIdsPtr_[id], commWeightsPtr_[id], ((GraphElem*)commIdKeys_[id]), ((GraphElem*)commIdKeys_[id])+ne, 
                mass_, v0, v1, e0, e1, V0, nv_per_device_);
        }
        //GraphElem m = MAX_GRIDDIM*BLOCKDIM02/WARPSIZE
        GraphWeight dq = compute_modularity_cuda(mod, num);
        #pragma omp critical
        q += dq;
        CudaFree(mod);
    }
    return q;
}

void GraphGPU::move_edges_to_device
(
    const GraphElem& e0,
    const GraphElem& e1,
    const int& host_id, 
    cudaStream_t stream
)
{
    if(e1 > e0)
    {
        if(e0 != e0_[host_id] or e1 != e1_[host_id])
        {
            GraphElem ne = e1-e0;
            e0_[host_id] = e0; e1_[host_id] = e1;
            CudaMemcpyAsyncHtoD(edges_[host_id], edgesHost_+e0, sizeof(GraphElem)*ne, stream);
        }
    }
}

void GraphGPU::move_edges_to_host
(
    const GraphElem& e0,  
    const GraphElem& e1,
    const int& host_id,
    cudaStream_t stream
)
{
    if(e1 > e0)
    {
        GraphElem ne = e1-e0;
        CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_[host_id], sizeof(GraphElem)*ne, stream);
    }
}

void GraphGPU::move_weights_to_device
(
    const GraphElem& e0, 
    const GraphElem& e1, 
    const int& host_id,
    cudaStream_t stream
)
{
    if(e1 > e0)
    {
        if(e0 != w0_[host_id] or e1 != w1_[host_id])
        {
            GraphElem ne = e1-e0;
            w0_[host_id] = e0; w1_[host_id] = e1;
            CudaMemcpyAsyncHtoD(edgeWeights_[host_id], edgeWeightsHost_+e0, sizeof(GraphWeight)*ne, stream);
        }
    }
}

void GraphGPU::move_weights_to_host
(
    const GraphElem& e0, 
    const GraphElem& e1, 
    const int& host_id,
    cudaStream_t stream
)
{
    if(e1 > e0)
    {
        GraphElem ne = e1-e0;
        CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_[host_id], sizeof(GraphWeight)*ne, stream);
    }
}

#ifdef MULTIPHASE
void GraphGPU::shuffle_edge_list()
{
    //update new ne
    NE_ = sortedIndicesHost_[NV_];

    GraphElem* bufferEdges = (GraphElem*)buffer_;
    #pragma omp parallel for
    for(Int i = 0; i < NV_; ++i)
    {
        GraphElem pos = vertexIdsHost_[i];
    
        GraphElem start  = indicesHost_ [pos];
        GraphElem start0 = sortedIndicesHost_[i+0];
        GraphElem num    = sortedIndicesHost_[i+1]-start0;
        for(GraphElem j = 0; j < num; ++j)
            bufferEdges[j+start0] = edgesHost_[start+j];
    }
    #pragma omp parallel for
    for(Int i = 0; i < NE_; ++i)
        edgesHost_[i] = bufferEdges[i];

    GraphWeight* bufferWeights = (GraphWeight*)buffer_;
    #pragma omp parallel for
    for(Int i = 0; i < NV_; ++i)
    {
        GraphElem pos = vertexIdsHost_[i];

        GraphElem start  = indicesHost_ [pos];
        GraphElem start0 = sortedIndicesHost_[i+0];
        GraphElem num    = sortedIndicesHost_[i+1]-start0;
        for(GraphElem j = 0; j < num; ++j)
            bufferWeights[start0+j] = edgeWeightsHost_[start+j];

    }

    #pragma omp parallel for
    for(Int i = 0; i < NE_; ++i)
        edgeWeightsHost_[i] = bufferWeights[i];
}

GraphElem GraphGPU::sort_vertex_by_community_ids()
{
    #pragma omp parallel
    {
        int i =  omp_get_thread_num() % NGPU;
        CudaSetDevice(i);

        GraphElem* vertexIds = newCommIds_[i];
        GraphElem V0 = v_base_[i];
        if(nv_[i] > 0)
        {    
            fill_vertex_index_cuda(vertexIds, nv_[i], V0);
    
            thrust::device_ptr<GraphElem> vertexIds_ptr = thrust::device_pointer_cast<GraphElem>(vertexIds);
            thrust::device_ptr<GraphElem> commIds_ptr   = thrust::device_pointer_cast<GraphElem>(commIds_[i]);

            thrust::stable_sort_by_key(commIds_ptr, commIds_ptr+nv_[i], vertexIds_ptr, thrust::less<GraphElem>());

            CudaMemcpyAsyncDtoH(commIdsHost_+V0,   commIds_[i], sizeof(GraphElem)*nv_[i], cuStreams[i][0]);
            CudaMemcpyAsyncDtoH(vertexIdsHost_+V0, vertexIds,   sizeof(GraphElem)*nv_[i], cuStreams[i][1]);

            CudaDeviceSynchronize();
        }
    }

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for
    for(GraphElem i = 0; i < NV_; ++i)
        sortedVertexIdsHost_[i] = {commIdsHost_[i], vertexIdsHost_[i]};

    less_int2 myless;

    std::stable_sort(sortedVertexIdsHost_, sortedVertexIdsHost_+NV_, myless)
   
    #pragma omp parallel for
    for(GraphElem i = 0; i < NV_; ++i)
    {
        GraphElem2 p = sortedVertexIdsHost_[i];
        commIdsHost_[p.y] = p.x;
        vertexIdsHost_[i] = p.y;
    } 

    GraphElem target = commIdsHost_[vertexIdsHost_[0]];
    GraphElem newNv = 0;
    for(GraphElem i = 0; i < NV_; ++i)
    {
        GraphElem myId = commIdsHost_[vertexIdsHost_[i]];
        if(target != myId)
        {
            vertexIdsOffsetHost_[newNv] = i;
            newNv++;
            target = myId;
        }    
    }
    for(GraphElem i = 0; i < newNv; ++i)
    {
        GraphElem start = 0;
        if(i > 0) start = vertexIdsOffsetHost_[i-1];
        GraphElem end = vertexIdsOffsetHost_[i];
        for(GraphElem j = start; j < end; ++j)
            commIdsHost_[vertexIdsHost_[j]] = i;
    }
    return newNv;
}

void GraphGPU::compress_all_edges()
{
    omp_set_num_threads(NGPU);
    #pragma omp parallel
    {
        int i = omp_get_thread_num() % NGPU;

        GraphElem start = indicesHost_[i+0];
        GraphElem end   = indicesHost_[i+1];

        std::unordered_map<GraphElem, GraphElem> edge_map; 
        GraphElem pos = 0;
        for(GraphElem j = start; j < end; ++j)
        {
            GraphElem myId = commIdsHost_[edgesHost_[j]];
            auto iter = edge_map.find(myId);
            if(iter != edge_map.end())
            {
                GraphElem u = iter->second;
                edgeWeightsHost_[start+u] += edgeWeightsHost_[j];
            }
            else
            {
                edge_map.insert({myId, pos});
                edgesHost_[start+pos] = myId;
                edgeWeightsHost_[start+pos] = edgeWeightsHost_[j];
                pos++;
            }
        }
        sortedIndicesHost_[i] = edge_map.size();
    }

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for
    for(GraphElem i = 0; i < NV_; ++i)
    {
        GraphElem pos = vertexIdsHost_[i];
        numEdgesHost_[i] = sortedIndicesHost_[pos];
    }
    sortedIndicesHost_[0] = 0;
    std::partial_sum(numEdgesHost_, numEdgesHost_+NV_, sortedIndicesHost_+1);
}

bool GraphGPU::aggregation()
{
    numEdgesHost_      =  (GraphElem*)sortedVertexIdsHost_;
    sortedIndicesHost_ = ((GraphElem*)sortedVertexIdsHost_)+NV_;

    GraphElem newNv = sort_vertex_by_community_ids();

    clusters_->update_clustering(commIdsHost_);

    if(newNv == NV_ || newNv == 1)
        return true;

    compress_all_edges();
    shuffle_edge_list();

    NV_ = newNv;
    #pragma omp parallel for
    for(GraphElem i = 0; i < NV_; ++i)
        indicesHost_[i+1] = sortedIndicesHost_[vertexIdsOffsetHost_[i]];

    nv_per_device_ = (NV_+NGPU-1)/NGPU;

    omp_set_num_threads(NGPU);
    #pragma omp parallel
    {
        int id = omp_get_thread_num() % NGPU;

        e0_[id] = 0; e1_[id] = 0;
        w0_[id] = 0; w1_[id] = 0;

        v_base_[id]  = nv_per_device_*id;
        if(v_base_[id] > NV_) v_base_[id] = NV_;

        v_end_ [id]  = v_base_[id]+nv_per_device_;
        if(v_end_[id] > NV_) v_end_[id] = NV_;

        e_base_[id] = indicesHost_[v_base_[id]];
        e_end_[id]  = indicesHost_[v_end_[id]];

        nv_[id] = v_end_[id]-v_base_[id];
        ne_[id] = e_end_[id]-e_base_[id];

        GraphElem nv = nv_[id];
        GraphElem ne = ne_[id];
        if(nv > 0)
        {
            CudaMemcpyAsyncHtoD(indices_[id], indicesHost_+v_base_[id], sizeof(GraphElem)*(nv+1), 0);
            CudaMemset(vertexWeights_, 0, sizeof(GraphWeight)*nv);
            CudaMemset(commWeights_, 0, sizeof(GraphWeight)*nv);
        } 
        unsigned unit_size = (sizeof(GraphElem) > sizeof(GraphWeight)) ? sizeof(GraphElem) : sizeof(GraphWeight);
        ne_per_partition_[id] = determine_optimal_edges_per_batch (nv, ne, unit_size);
        GraphElem ne_per_partition = ne_per_partition_[id];

        vertex_partition_[id].clear();
        determine_optimal_vertex_partition(indicesHost_, nv, ne, ne_per_partition, vertex_partition_[id], v_base_[id]);
        sum_vertex_weights(id);
    }

    compute_mass();
    std::cout << mass_ << std::endl;

    maxOrder_ = max_order();
    //std::cout << "max order is " << maxOrder_ << std::endl;
    if(maxOrder_ > ne_per_partition_)
    {
        std::cout << "max order is " << maxOrder_ << std::endl;
        std::cout << "vertex order is too large" << std::endl;
        exit(-1);
    }

    maxPartitions_ = vertex_partition_[0].size();
    for(int i = 1; i < NGPU; ++i)
    {
        if(vertex_partition_[i].size() > maxPartitions_)
            maxPartitions_ = vertex_partition_[i].size();
    }

    return false;
}

void GraphGPU::dump_partition(const std::string& filename)
{
    clusters_->dump_partition(filename);
}
#endif
