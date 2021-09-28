#include <thrust/sort.h>
#include <thrust/execution_policy.h>
//#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <omp.h>
#include <algorithm>
#include <unordered_map>
#include "graph_gpu.hpp"
#include "graph_cuda.hpp"
//include host side definition of graph
#include "graph.hpp"
#include "cuda_wrapper.hpp"

GraphGPU::GraphGPU(Graph* graph, const int& nbatches, const int& part_on_device, const int& part_on_batch) : 
graph_(graph), nbatches_(nbatches), part_on_device_(part_on_device), part_on_batch_(part_on_batch), 
NV_(0), NE_(0), maxOrder_(0), mass_(0)
{
    NV_ = graph_->get_num_vertices();
    NE_ = graph_->get_num_edges();

    unsigned unit_size = (sizeof(GraphElem) > sizeof(GraphWeight)) ? sizeof(GraphElem) : sizeof(GraphWeight);       
    indicesHost_     = graph_->get_index_ranges();
    edgeWeightsHost_ = graph_->get_edge_weights();
    edgesHost_       = graph_->get_edges();

    determine_edge_device_partition();

    omp_set_num_threads(NGPU);
    #pragma omp parallel
    {
        int id =  omp_get_thread_num() % NGPU;   
        CudaSetDevice(id);

        for(unsigned i = 0; i < 4; ++i)
            CudaCall(cudaStreamCreate(&cuStreams[id][i]));

        e0_[id] = 0; e1_[id] = 0;
        w0_[id] = 0; w1_[id] = 0;

        GraphElem nv = nv_[id];
        GraphElem ne = ne_[id];

        CudaMalloc(indices_[id],       sizeof(GraphElem)   *(nv+1));
        CudaMalloc(vertexWeights_[id], sizeof(GraphWeight) *nv);
        CudaMalloc(commIds_[id],       sizeof(GraphElem)   *nv);
        CudaMalloc(commWeights_[id],   sizeof(GraphWeight) *nv);
        CudaMalloc(newCommIds_[id],    sizeof(GraphElem)   *nv);
        CudaMalloc(commIdsPtr_[id],    sizeof(GraphElem*)  *NGPU);
        CudaMalloc(commWeightsPtr_[id],sizeof(GraphWeight*)*NGPU);

        CudaMalloc(vertex_per_device_[id], sizeof(GraphElem)*(NGPU+1));

        CudaMemcpyAsyncHtoD(indices_[id], indicesHost_+v_base_[id], sizeof(GraphElem)*(nv+1), 0);
        CudaMemcpyAsyncHtoD(vertex_per_device_[id], vertex_per_device_host_, sizeof(GraphElem)*(NGPU+1),0);

        CudaMemset(vertexWeights_[id], 0, sizeof(GraphWeight)*nv);
        CudaMemset(commWeights_[id],   0, sizeof(GraphWeight)*nv);

        ne_per_partition_[id] = determine_optimal_edges_per_partition(nv, ne, unit_size);
        GraphElem ne_per_partition = ne_per_partition_[id];

        determine_optimal_vertex_partition(indicesHost_, nv, ne, ne_per_partition, vertex_partition_[id], v_base_[id]);

        CudaMalloc(edges_[id],          unit_size*ne_per_partition);
        CudaMalloc(edgeWeights_[id],    unit_size*ne_per_partition);
        CudaMalloc(commIdKeys_[id],     sizeof(GraphElem2)*ne_per_partition);
        //CudaMalloc(reducedWeights_[id], sizeof(GraphWeight)*ne_per_partition);
        CudaMalloc(orderedWeights_[id], sizeof(GraphWeight)*ne_per_partition);

        CudaMalloc(localCommNums_[id],    sizeof(GraphElem)*(nv+1));
        CudaMemset(localCommNums_[id], 0, sizeof(GraphElem)*(nv+1));

        //CudaMalloc(indexOrders_[id],  sizeof(GraphElem)*ne_per_partition);
        //CudaMalloc(localCommNums_[id],sizeof(GraphElem)*nv);
        //CudaMalloc(localOffsets_[id], sizeof(GraphElem)*ne_per_partition);

        CudaCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

        ordered_weights_ptr[id] = thrust::device_pointer_cast(orderedWeights_[id]);
        //reduced_weights_ptr[id] = thrust::device_pointer_cast(orderedWeights_[id]);
        keys_ptr[id]            = thrust::device_pointer_cast(commIdKeys_[id]);
        local_comm_nums_ptr[id] = thrust::device_pointer_cast(localCommNums_[id]);

        sum_vertex_weights(id);
        CudaDeviceSynchronize();

        vertex_per_batch_[id] = new GraphElem [nbatches+1];
        vertex_per_batch_partition_[id].resize(nbatches);
        //CudaDeviceSynchronize();
    }
    maxOrder_ = max_order();
    for(int i = 0; i < NGPU; ++i)
    {
        if(maxOrder_ > ne_per_partition_[i])
        {
            std::cout <<"Maximum order exceeds the maximum edge capacity\n";
            exit(-1);
        }
    }
    //std::cout << "max order is " << maxOrder_ << std::endl;
    compute_mass();

    #ifdef MULTIPHASE
    buffer_ = malloc(unit_size*NE_);
    commIdsHost_ = new GraphElem [NV_];
    vertexIdsHost_ = new GraphElem [NV_];
    vertexIdsOffsetHost_ = new GraphElem [NV_];
    sortedVertexIdsHost_ = (GraphElem2*)malloc(sizeof(GraphElem2)*(NV_+1));

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

    partition_graph_edge_batch();
/*
    std::cout << "total V: " << NV_ << " total E: " << NE_ << std::endl;
    for(int i = 0; i < NGPU; ++i)
    {
        std::cout << nv_[i] << " " << ne_[i] << " " << ne_per_partition_[i] << std::endl;
    }
    for(int i = 0; i <= NGPU; ++i)
        std::cout << i << " " << vertex_per_device_host_[i] << std::endl;
    std::cout << std::endl;
    for(int i = 0; i < NGPU; ++i)
    {
        for(int j = 0; j < nbatches_+1; ++j)
            std::cout << vertex_per_batch_[i][j] << std::endl;
        std::cout << std::endl;
    }
    for(int i = 0; i < NGPU; ++i)
    {
        for(int j = 0; j < nbatches_; ++j)
        {
            std::vector<GraphElem>& vec = vertex_per_batch_partition_[i][j];
            for(int k = 0; k < vec.size(); ++k)
            {
                 GraphElem id = vertex_per_batch_partition_[i][j][k];
                 std::cout << id << " " << indicesHost_[id] << std::endl;
            } 
        }
        std::cout << std::endl;
    }
    for(int i = 0; i < NGPU; ++i)
    {
        for(int j = 0; j < vertex_partition_[i].size(); ++j)
            std::cout << vertex_partition_[i][j] << std::endl;
        std::cout << std::endl;
    }*/ 
    //exit(-1);
    //maxPartitions_ = vertex_partition_[0].size();
    //ne_per_partition_cap_ = ne_per_partition_[0];
/*
    for(int i = 1; i < NGPU; ++i)
    {
        if(vertex_partition_[i].size() > maxPartitions_)
            maxPartitions_ = vertex_partition_[i].size();
        if(ne_per_partition_[i] < ne_per_partition_cap_)
            ne_per_partition_cap_ = ne_per_partition_[i];
    }*/
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
        CudaFree(localCommNums_[g]);
        CudaFree(orderedWeights_[g]);
        CudaFree(vertex_per_device_[g]);
        //CudaFree(reducedWeights_[g]);
        delete [] vertex_per_batch_[g];
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

void GraphGPU::determine_edge_device_partition()
{
    std::vector<GraphElem> vertex_parts;
    vertex_parts.push_back(0);

    if(!part_on_device_)
    {
        GraphElem ave_nv = NV_/NGPU;
        for(int i = 1; i < NGPU; ++i)
            vertex_parts.push_back(i*ave_nv);
        vertex_parts.push_back(NV_);
    }
    else
    {
        GraphElem start = 0;
        GraphElem ave_ne = NE_/NGPU;
        for(GraphElem i = 1; i <= NV_; ++i)
        { 
            if(indicesHost_[i]-indicesHost_[start] > ave_ne)
            {
                vertex_parts.push_back(i);
                start = i;
            }
            else if (i == NV_)
                vertex_parts.push_back(NV_);
        }

        if(vertex_parts.size() > NGPU+1)
        {
            GraphElem remain = NV_ - vertex_parts[NGPU];
            GraphElem nv = remain/NGPU;
            for(GraphElem i = 1; i < NGPU; ++i)
                vertex_parts[i] += nv;
            vertex_parts[NGPU] = NV_;
        }
        else if(vertex_parts.size() < NGPU+1)
        {
            for(int i = vertex_parts.size(); i < NGPU+1; ++i)
                vertex_parts.push_back(NV_);
        }
    }
    for(int i = 0; i <= NGPU; ++i)
        vertex_per_device_host_[i] = vertex_parts[i];
    
    for(GraphElem id = 0; id < NGPU; ++id)
    {
        v_base_[id] = vertex_parts[id+0];
        v_end_ [id] = vertex_parts[id+1];

        e_base_[id] = indicesHost_[v_base_[id]];
        e_end_[id]  = indicesHost_[v_end_[id]];

        nv_[id] = v_end_[id]-v_base_[id];
        ne_[id] = e_end_[id]-e_base_[id];
    }
}

void GraphGPU::partition_graph_edge_batch()
{ 
    for(int g= 0; g < NGPU; ++g)
    {
        GraphElem nv = nv_[g]; 
        GraphElem ne = ne_[g];
        std::vector<GraphElem> vertex_parts;
        GraphElem V0 = v_base_[g];

        if(!part_on_batch_)
        {
            vertex_parts.resize(nbatches_+1);
            GraphElem ave_nv = nv / nbatches_;
            for(int i = 0; i < nbatches_; ++i)
            {
                vertex_parts[i] = (GraphElem)i*ave_nv+V0;
                if(vertex_parts[i] > nv)
                    vertex_parts[i] = nv+V0;
            }
            vertex_parts[nbatches_] = nv+V0; 
        }
        else
        {
            GraphElem ave_ne = ne / nbatches_;
            /*if(ave_ne == 0)
            {
                std::cout << "Too many batches\n";
                exit(-1); 
            }*/

            vertex_parts.push_back(V0);

            GraphElem start = V0;
            for(GraphElem i = 1; i <= nv; ++i)
            {
                if(indicesHost_[i+V0]-indicesHost_[start] > ave_ne)
                {
                    vertex_parts.push_back(i+V0);
                    start = i+V0;
                }
                else if (i == nv)
                    vertex_parts.push_back(v_end_[g]);
            }

            if(vertex_parts.size() > nbatches_+1)
            {
                GraphElem remain = v_end_[g] - vertex_parts[nbatches_];
                GraphElem nnv = remain/nbatches_;
                for(int i = 1; i < nbatches_; ++i)
                    vertex_parts[i] += nnv;
                vertex_parts[nbatches_] = v_end_[g];
            }
            else if(vertex_parts.size() < nbatches_+1) 
            {
                for(int i = vertex_parts.size(); i < nbatches_+1; ++i)
                    vertex_parts.push_back(v_end_[g]);
            }
        }
        for(int i = 0; i < nbatches_+1; ++i)
            vertex_per_batch_[g][i] = vertex_parts[i];
  
        for(int b = 0; b < nbatches_; ++b)
        {
            GraphElem v0 = vertex_per_batch_[g][b+0];
            GraphElem v1 = vertex_per_batch_[g][b+1];
            GraphElem start = v0; 
            vertex_per_batch_partition_[g][b].push_back(v0);
            for(GraphElem i = v0+1; i <= v1; ++i)
            {
                if(indicesHost_[i]-indicesHost_[start] > ne_per_partition_[g])
                {
                    vertex_per_batch_partition_[g][b].push_back(i-1);
                    start = i-1;
                    i--;
                }  
            }
            vertex_per_batch_partition_[g][b].push_back(v1);
        }    
    }
}

//TODO: in the future, this could be moved to a new class
GraphElem GraphGPU::determine_optimal_edges_per_partition
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

        float occ_m = (uint64_t)((4*sizeof(GraphElem)+2*sizeof(GraphWeight))*nv)/1048576.0;
        free_m =(uint64_t)free_t/1048576.0 - occ_m;

        GraphElem ne_per_partition = (GraphElem)(free_m / unit_size / 8 * 1048576.0); //5 is the minimum, i chose 8
        //std::cout << ne_per_partition << " " << ne << std::endl;
        #ifdef debug
        return ((ne_per_partition > ne) ? ne : ne_per_partition);
        #else 
        return ((ne_per_partition > ne) ? ne : ne_per_partition);
        //return ne/4;
        #endif
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
    GraphElem start = indices[V0];
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

#ifdef CHECK

void GraphGPU::set_community_ids()
{
    GraphElem* commIds = new GraphElem [NV_];
    for(GraphElem i = 0; i < NV_; ++i)
        commIds[i] = rand()%2;
    set_community_ids(commIds);

    delete [] commIds;
}

#endif

void GraphGPU::set_community_ids(GraphElem* commIds)
{
    for(int i = 0; i < NGPU; ++i) 
    {
        if(nv_[i] > 0)
        {
            CudaSetDevice(i);
            CudaMemcpyAsyncHtoD(commIds_[i],    commIds+v_base_[i], sizeof(GraphElem)*nv_[i], cuStreams[i][0]);
            CudaMemcpyAsyncHtoD(newCommIds_[i], commIds+v_base_[i], sizeof(GraphElem)*nv_[i], cuStreams[i][1]);

            compute_community_weights(i);

            CudaDeviceSynchronize();
        }
    }
}

void GraphGPU::compute_community_weights(const int& host_id)
{
    if(nv_[host_id] > 0)
        compute_community_weights_cuda(commWeightsPtr_[host_id], commIds_[host_id], 
        vertexWeights_[host_id], nv_[host_id], v_base_[host_id], vertex_per_device_[host_id]);
}

void GraphGPU::sort_edges_by_community_ids
(
    const GraphElem& v0,   //starting global vertex index 
    const GraphElem& v1,   //ending global vertex index
    const GraphElem& e0,   //starting global edge index
    const GraphElem& e1,   //ending global edge index
    const int& host_id,
    const int& exclude_self_loops
)
{
    if(v1 > v0)
    {
        GraphElem ne = e1-e0;
        GraphElem V0 = v_base_[host_id];

        GraphElem e0_local = e0 - e0_[host_id];
        GraphElem w0_local = e0 - w0_[host_id]; 
        GraphElem nv = v1 - v0;
 
        fill_edges_community_ids_cuda(commIdKeys_[host_id], localCommNums_[host_id]+1, edges_[host_id]+e0_local, 
        indices_[host_id], commIdsPtr_[host_id],  v0, v1, e0, e1, V0, vertex_per_device_[host_id], cuStreams[host_id][0]);

        copy_weights_cuda(orderedWeights_[host_id], edgeWeights_[host_id]+w0_local, edges_[host_id]+e0_local, 
        indices_[host_id], v0, v1, e0, e1, V0, exclude_self_loops, cuStreams[host_id][1]);

        //CudaDeviceSynchronize();

        thrust::stable_sort_by_key(keys_ptr[host_id], keys_ptr[host_id]+ne, ordered_weights_ptr[host_id], comp);

        thrust::pair<thrust::device_ptr<GraphElem2>,thrust::device_ptr<GraphWeight> > new_ends;

        new_ends = thrust::reduce_by_key(keys_ptr[host_id], keys_ptr[host_id]+ne, ordered_weights_ptr[host_id], 
        keys_ptr[host_id], ordered_weights_ptr[host_id], is_equal_int2);

        GraphElem num_unique_id = new_ends.first - keys_ptr[host_id];

        fill_unique_community_counts_cuda(localCommNums_[host_id]+1, commIdKeys_[host_id], v0, num_unique_id);

        //CudaDeviceSynchronize();

        thrust::inclusive_scan(local_comm_nums_ptr[host_id], local_comm_nums_ptr[host_id]+nv+1, local_comm_nums_ptr[host_id]);
    }
}

void GraphGPU::singleton_partition()
{
    omp_set_num_threads(NGPU);
    //std::cout << NGPU << std::endl;
    #pragma omp parallel
    //for(int i = 0; i < NGPU; ++i)
    {
        //std::cout << omp_get_num_threads() << std::endl;
        int i = omp_get_thread_num() % NGPU;
        //std::cout << i << std::endl;
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
    omp_set_num_threads(NGPU);
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
        GraphElem v0 = vertex_partition_[host_id][b+0];
        GraphElem v1 = vertex_partition_[host_id][b+1];


        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];

        move_weights_to_device(e0, e1, host_id);

        if(v1 > v0)
            sum_vertex_weights_cuda(vertexWeights_[host_id], edgeWeights_[host_id], indices_[host_id], v0, v1, e0, e1, V0);
    }
}

void GraphGPU::louvain_update
(
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    //const GraphElem& e0_local,
    const int& host_id
)
{
    if(v1 > v0)
    {
        //GraphElem ne = e1-e0;
        GraphElem V0 = v_base_[host_id];
        //GraphElem e0_local = e0 - e0_[host_id];
        //GraphElem w0_local = e0 - w0_[host_id];
        //louvain_update_cuda(((GraphElem*)commIdKeys_[host_id]), ((GraphElem*)commIdKeys_[host_id])+ne, 
        louvain_update_cuda(commIdKeys_[host_id], localCommNums_[host_id], orderedWeights_[host_id], 
                            vertexWeights_[host_id], commIds_[host_id], commWeightsPtr_[host_id], newCommIds_[host_id], 
                            mass_, v0, v1, e0, e1, V0, vertex_per_device_[host_id]);
    }
}    

void GraphGPU::louvain_update
(
    const int& bb, 
    const int& host_id
)
{
    const std::vector<GraphElem>& batch = vertex_per_batch_partition_[host_id][bb];
    for(int b = 0; b < batch.size()-1; ++b)
    {
        GraphElem v0 = batch[b+0];
        GraphElem v1 = batch[b+1];
        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];

        move_edges_to_device(e0, e1, host_id, cuStreams[host_id][1]);

        move_weights_to_device(e0, e1, host_id, cuStreams[host_id][2]);

        sort_edges_by_community_ids(v0, v1, e0, e1, host_id, 1);

        louvain_update(v0, v1, e0, e1, host_id);
    }
}
#ifdef CHECK
void GraphGPU::louvain_update_host
(
    const int& batch
)
{
    GraphElem*   commIds         = new GraphElem[NV_];
    GraphElem*   newCommIds      = new GraphElem[NV_];
    GraphWeight* ac              = new GraphWeight[NV_];
    GraphWeight* weighted_orders = new GraphWeight[NV_];

    GraphWeight** e_cj = new GraphWeight* [16];
    for(int i = 0; i < 16; ++i) 
        e_cj[i] = new GraphWeight [maxOrder_];
 
    for(int g = 0; g < NGPU; ++g)
    {
        CudaSetDevice(g);
        CudaMemcpyAsyncDtoH(commIds+v_base_[g], commIds_[g], nv_[g]*sizeof(GraphElem),  cuStreams[g][0]);
        CudaMemcpyAsyncDtoH(newCommIds+v_base_[g], commIds_[g], nv_[g]*sizeof(GraphElem),  cuStreams[g][1]);   
        CudaMemcpyAsyncDtoH(ac+v_base_[g], commWeights_[g], nv_[g]*sizeof(GraphWeight), cuStreams[g][2]);
        CudaMemcpyAsyncDtoH(weighted_orders+v_base_[g], vertexWeights_[g], nv_[g]*sizeof(GraphWeight), cuStreams[g][3]);
    }
    for(int g = 0;  g < NGPU; ++g)
    {
        CudaSetDevice(g);
        CudaDeviceSynchronize();
    }

    for(int g = 0; g < NGPU; ++g)
    {
        GraphElem v0 = vertex_per_batch_[g][batch+0];
        GraphElem v1 = vertex_per_batch_[g][batch+1];

        for(int n = 0; n < 16; ++n)
            for(GraphElem i = 0; i < maxOrder_; ++i)
                e_cj[n][i] = 0;

        omp_set_num_threads(16);
        #pragma omp parallel for
        for(GraphElem v = v0; v < v1; ++v)
        {
            GraphElem   *edges     = edgesHost_+indicesHost_[v];
            GraphElem    num_edges = indicesHost_[v+1]-indicesHost_[v];
            GraphWeight *weights   = edgeWeightsHost_+indicesHost_[v];
            GraphElem my_comm_id = commIds[v];
            int my_thread_id = omp_get_thread_num();

            //loop through all neighboring clusters
            GraphElem unique_id = 0;
            std::unordered_map<GraphElem, GraphElem> neighCommIdMap;

            GraphWeight e_ci = 0.;
            GraphWeight ki = weighted_orders[v];
            for(GraphElem j = 0; j < num_edges; ++j)
            {
                GraphElem u      = edges[j];
                GraphWeight w_vu = weights[j];

                //Int comm_id = partition_->get_comm_id(u);
                GraphElem comm_id = commIds[u];
                if(comm_id != my_comm_id)
                {
                    std::unordered_map<Int,Int>::iterator iter;
                    if((iter = neighCommIdMap.find(comm_id)) == neighCommIdMap.end())
                    {
                        neighCommIdMap.insert({comm_id, unique_id});
                        e_cj[my_thread_id][unique_id] = w_vu; //graph->get_weight(vi,j);
                        unique_id++;
                    }
                    else
                        e_cj[my_thread_id][iter->second] += w_vu;
                }
                else if(v != u)
                    e_ci += w_vu;
            }

            //determine the best move
            GraphWeight ac_i;
            ac_i = ac[my_comm_id]-ki;

            GraphWeight delta = 0;
            GraphElem destCommId = my_comm_id;

            for(auto iter = neighCommIdMap.begin(); iter != neighCommIdMap.end(); ++iter)
            {
                GraphWeight val = e_cj[my_thread_id][iter->second];
                val -= (e_ci - ki*(ac_i-ac[iter->first])/(2.*mass_));
                val /= mass_;
                if(val > delta)
                {
                    destCommId = iter->first;
                    delta = val;
                }
            }
            if(destCommId != my_comm_id)
                newCommIds[v] = destCommId;
            else 
                newCommIds[v] = my_comm_id;
        }
    }
    for(int g = 0; g < NGPU; ++g)
    {
        CudaSetDevice(g);
        CudaMemcpyAsyncDtoH(commIds+v_base_[g], newCommIds_[g], nv_[g]*sizeof(GraphElem),  cuStreams[g][0]);
    }
    for(int g = 0; g < NGPU; ++g)
    {
        CudaSetDevice(g);
        CudaDeviceSynchronize();
    }

    GraphElem count = 0;
    for(int g = 0; g < NGPU; ++g)
    {
        GraphElem v0 = vertex_per_batch_[g][batch+0];
        GraphElem v1 = vertex_per_batch_[g][batch+1];
        for(GraphElem i = v0; i < v1; ++i) 
        {
            if(newCommIds[i]!=commIds[i])
                count++;
            if(i < 10)
                std::cout << newCommIds[i] << " " << commIds[i] << std::endl;
        }
    }
    std::cout << count << std::endl;
    for(int i = 0; i < 16; ++i) 
        delete e_cj[i];
    delete e_cj;

    delete [] commIds;
    delete [] newCommIds;
    delete [] ac;
    delete [] weighted_orders;
}
#endif

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
                                      vertexWeights_[host_id]+v0-V0, nv, vertex_per_device_[host_id]);
 
}

void GraphGPU::update_community_weights
(
    const int& batch,
    const int& host_id
)
{
    GraphElem v0 = vertex_per_batch_[host_id][batch+0];
    GraphElem v1 = vertex_per_batch_[host_id][batch+1];
    GraphElem e0 = indicesHost_[v0];
    GraphElem e1 = indicesHost_[v1];

    update_community_weights(v0, v1, e0, e1, host_id); 
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

void GraphGPU::update_community_ids
(
    const int& batch,
    const int& host_id
)
{
    GraphElem v0 = vertex_per_batch_[host_id][batch+0];
    GraphElem v1 = vertex_per_batch_[host_id][batch+1];
    GraphElem e0 = indicesHost_[v0];
    GraphElem e1 = indicesHost_[v1];

    update_community_ids(v0, v1, e0, e1, host_id);
}

void GraphGPU::restore_community() 
{
    for(int i = 0; i < NGPU; ++i)
    {
        if(nv_[i] > 0)
        {
            CudaSetDevice(i);
            update_community_weights_cuda(commWeightsPtr_[i], commIds_[i], newCommIds_[i], vertexWeights_[i], nv_[i], vertex_per_device_[i]);
            CudaDeviceSynchronize();
        }
    }
    omp_set_num_threads(NGPU);
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
    omp_set_num_threads(NGPU);
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
    //std::cout << mass_ << std::endl;
}

GraphWeight GraphGPU::compute_modularity()
{
    GraphWeight q = 0.;
    omp_set_num_threads(NGPU);
    #pragma omp parallel
    {
        int id = omp_get_thread_num() % NGPU;
        CudaSetDevice(id);
 
        GraphWeight* mod;
        GraphElem num = MAX_GRIDDIM*BLOCKDIM02/TILESIZE02;

        CudaMalloc(mod,    sizeof(GraphWeight)*num);
        CudaMemset(mod, 0, sizeof(GraphWeight)*num);
        GraphElem V0 = v_base_[id];
        //std::cout << vertex_partition_[id].size()-1 << std::endl;
        for(GraphElem b = 0; b < vertex_partition_[id].size()-1; ++b)
        {
            GraphElem v0 = vertex_partition_[id][b];
            GraphElem v1 = vertex_partition_[id][b+1];

            GraphElem e0 = indicesHost_[v0];
            GraphElem e1 = indicesHost_[v1];
            
            move_edges_to_device(e0, e1, id, cuStreams[id][1]);
            
            move_weights_to_device(e0, e1, id, cuStreams[id][2]);

            /*sort_edges_by_community_ids(v0, v1, e0, e1, id, 0);
            
            if(v1 > v0) 
                compute_modularity_reduce_cuda<BLOCKDIM02, WARPSIZE>(mod, commIdKeys_[id], localCommNums_[id], orderedWeights_[id], 
                commIds_[id], commWeightsPtr_[id], mass_, v0, v1, e0, e1, V0, vertex_per_device_[id]);*/
            compute_modularity_reduce_cuda<BLOCKDIM02, TILESIZE02>(mod, indices_[id], edges_[id] , edgeWeights_[id], 
            commIds_[id], commIdsPtr_[id], commWeightsPtr_[id], mass_, v0, v1, e0, e1, V0, vertex_per_device_[id]);
        }
        //GraphElem m = MAX_GRIDDIM*BLOCKDIM02/WARPSIZE
        GraphWeight dq = compute_modularity_cuda(mod, num);
        #pragma omp critical
        q += dq;
        CudaFree(mod);
    }
    return q;
}

#ifdef CHECK
void GraphGPU::compute_modularity_host()
{
    GraphElem*   commIds         = new GraphElem[NV_];
    GraphWeight* ac              = new GraphWeight[NV_];
    GraphWeight* weighted_orders = new GraphWeight[NV_];

    for(int g = 0; g < NGPU; ++g)
    {
        CudaSetDevice(g);
        CudaMemcpyAsyncDtoH(commIds+v_base_[g], commIds_[g], nv_[g]*sizeof(GraphElem),  cuStreams[g][0]);
        CudaMemcpyAsyncDtoH(ac+v_base_[g], commWeights_[g], nv_[g]*sizeof(GraphWeight), cuStreams[g][2]);
        CudaMemcpyAsyncDtoH(weighted_orders+v_base_[g], vertexWeights_[g], nv_[g]*sizeof(GraphWeight), cuStreams[g][3]);
    }
    for(int g = 0;  g < NGPU; ++g)
    {
        CudaSetDevice(g);
        CudaDeviceSynchronize();
    }

    GraphWeight q = 0.;
    for(int g = 0; g < NGPU; ++g)
    {
        GraphElem v0 = v_base_[g];
        GraphElem v1 = v_end_[g];

        omp_set_num_threads(16);
        #pragma omp parallel for
        for(GraphElem v = v0; v < v1; ++v)
        {
            GraphElem   *edges     = edgesHost_+indicesHost_[v];
            GraphElem    num_edges = indicesHost_[v+1]-indicesHost_[v];
            GraphWeight *weights   = edgeWeightsHost_+indicesHost_[v];
            GraphElem my_comm_id = commIds[v];

            //loop through all neighboring clusters
            for(GraphElem j = 0; j < num_edges; ++j)
            {
                GraphElem u      = edges[j];
                GraphWeight w_vu = weights[j];

                //Int comm_id = partition_->get_comm_id(u);
                GraphElem comm_id = commIds[u];
                if(comm_id == my_comm_id)
                {
                    #pragma omp critical
                    q += w_vu/(2.*mass_);
                }
            }
            #pragma omp critical
            q-=ac[v]*ac[v]/(4.*mass_*mass_);
        }
    }
    std::cout << "Modularity on CPU: " << q << std::endl;

    delete [] commIds;
    delete [] ac;
    delete [] weighted_orders;
}
#endif
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
        if(e0 < e0_[host_id] or e1 > e1_[host_id])
        {
            e0_[host_id] = e0; e1_[host_id] = e0+ne_per_partition_[host_id];
            if(e1_[host_id] > e_end_[host_id])
                e1_[host_id] = e_end_[host_id];
            if(e1_[host_id] < e1)
                std::cout << "range error\n";
            GraphElem ne = e1_[host_id]-e0_[host_id];
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
        if(e0 < w0_[host_id] or e1 > w1_[host_id])
        {
            w0_[host_id] = e0; w1_[host_id] = e0+ne_per_partition_[host_id];
            if(w1_[host_id] > e_end_[host_id])
                w1_[host_id] = e_end_[host_id];
            if(w1_[host_id] < e1)
                std::cout << "range error\n";

            GraphElem ne = w1_[host_id] - w0_[host_id];
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
/*
void GraphGPU::shuffle_edge_list()
{
    //update new ne
    NE_ = sortedIndicesHost_[NV_];
    GraphElem* bufferEdges = (GraphElem*)buffer_;

    omp_set_num_threads(omp_get_max_threads());
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
*/

void GraphGPU::shuffle_edge_list()
{
    omp_set_num_threads(omp_get_max_threads());
    /*#pragma omp parallel for 
    for(GraphElem i = 0; i < NV_; ++i)
    {
        GraphElem start = indicesHost_[i+0];
        GraphElem end   = indicesHost_[i+1];

        sortedIndicesHost_[i] = end-start; //edge_map.size();
    }*/

    #pragma omp parallel for 
    for(GraphElem i = 0; i < NV_; ++i)
    {
        GraphElem pos = vertexIdsHost_[i]; 
        //numEdgesHost_[i] = sortedIndicesHost_[pos];
        numEdgesHost_[i] = indicesHost_[pos+1]-indicesHost_[pos+0];
    }
    sortedIndicesHost_[0] = 0;
    std::partial_sum(numEdgesHost_, numEdgesHost_+NV_, sortedIndicesHost_+1);

    GraphElem* bufferEdges = (GraphElem*)buffer_;

    omp_set_num_threads(omp_get_max_threads());
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
    omp_set_num_threads(NGPU);
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

    std::stable_sort(sortedVertexIdsHost_, sortedVertexIdsHost_+NV_, myless);
   
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
    vertexIdsOffsetHost_[newNv] = NV_;
    newNv++;
    #pragma omp parallel
    for(GraphElem i = 0; i < newNv; ++i)
    {
        GraphElem start = 0;
        if(i > 0) start = vertexIdsOffsetHost_[i-1];
        GraphElem end = vertexIdsOffsetHost_[i];
        for(GraphElem j = start; j < end; ++j)
            commIdsHost_[vertexIdsHost_[j]] = i;
    }
    //for(GraphElem i = 0; i < newNv; ++i)
    //    std::cout <<  vertexIdsOffsetHost_[i] << std::endl; 
    return newNv;
}

void GraphGPU::compress_edges()
{
    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for
    for(GraphElem i = 0; i < NV_; ++i)
    {
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
        numEdgesHost_[i] = pos; //edge_map.size();
    }

    sortedIndicesHost_[0] = 0;
    std::partial_sum(numEdgesHost_, numEdgesHost_+NV_, sortedIndicesHost_+1);
    //std::inclusive_scan(std::execution::par, numEdgesHost_, numEdgesHost_+NV_, sortedIndicesHost_+1);
    //parallel_inclusive_scan(numEdgesHost_, numEdgesHost_+NV_, sortedIndicesHost_+1);
 
    //update number of edges    
    NE_ = sortedIndicesHost_[NV_]; 

    GraphElem* bufferEdges = (GraphElem*)buffer_;

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for
    for(Int i = 0; i < NV_; ++i)
    {
        GraphElem start  = indicesHost_[i];
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
        GraphElem start  = indicesHost_[i];
        GraphElem start0 = sortedIndicesHost_[i+0];
        GraphElem num    = sortedIndicesHost_[i+1]-start0;
        for(GraphElem j = 0; j < num; ++j)
            bufferWeights[start0+j] = edgeWeightsHost_[start+j];
    }

    #pragma omp parallel for
    for(Int i = 0; i < NE_; ++i)
        edgeWeightsHost_[i] = bufferWeights[i];

    #pragma omp parallel for
    for(Int i = 0; i < NV_+1; ++i)
        indicesHost_[i] = sortedIndicesHost_[i];

    //for(GraphElem i = 0; i < NV_; ++i)
    //    std::cout << sortedIndicesHost_[i] << " " << sortedIndicesHost_[i+1] << std::endl;
}
/*
void GraphGPU::compress_all_edges()
{
    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for
    for(GraphElem i = 0; i < NV_; ++i)
    {
        GraphElem start = indicesHost_[i+0];
        GraphElem end   = indicesHost_[i+1];

        std::unordered_map<GraphElem, GraphElem> edge_map; 
        GraphElem pos = 0;
        for(GraphElem j = start; j < end; ++j)
        {
            GraphElem myId = edgesHost_[j];
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
        numEdgesHost_[i] = pos; //edge_map.size();
    }
    sortedIndicesHost_[0] = 0;
    std::partial_sum(numEdgesHost_, numEdgesHost_+NV_, sortedIndicesHost_+1);

    //std::cout << NE_ << std::endl;
    //std::cout <<  sortedIndicesHost_[NV_] << " " << indicesHost_[NV_] << std::endl;
    //NE_ = indicesHost_[NV_];
    NE_ = sortedIndicesHost_[NV_];
    //std::cout << NE_ << std::endl;

    GraphElem* bufferEdges = (GraphElem*)buffer_;

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for
    for(Int i = 0; i < NV_; ++i)
    {
        GraphElem start  = indicesHost_[i];
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
        GraphElem start  = indicesHost_[i];
        GraphElem start0 = sortedIndicesHost_[i+0];
        GraphElem num    = sortedIndicesHost_[i+1]-start0;
        for(GraphElem j = 0; j < num; ++j)
            bufferWeights[start0+j] = edgeWeightsHost_[start+j];
    }

    #pragma omp parallel for
    for(Int i = 0; i < NE_; ++i)
        edgeWeightsHost_[i] = bufferWeights[i];

    #pragma omp parallel for
    for(Int i = 0; i < NV_+1; ++i)
        indicesHost_[i] = sortedIndicesHost_[i];
    
}
*/
bool GraphGPU::aggregation()
{
    numEdgesHost_      =  (GraphElem*)sortedVertexIdsHost_;
    sortedIndicesHost_ = ((GraphElem*)sortedVertexIdsHost_)+NV_;

    GraphElem newNv = sort_vertex_by_community_ids();

    clusters_->update_clustering(commIdsHost_);

    if(newNv == NV_ || newNv == 1)
        return true;

    shuffle_edge_list();
    
    NV_ = newNv;
    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for
    for(GraphElem i = 0; i < NV_; ++i)
        indicesHost_[i+1] = sortedIndicesHost_[vertexIdsOffsetHost_[i]];

    compress_edges();

    //shuffle_edge_list();

/*    NV_ = newNv;

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for
    for(GraphElem i = 0; i < NV_; ++i)
        indicesHost_[i+1] = sortedIndicesHost_[vertexIdsOffsetHost_[i]];
*/
    //compress_all_edges();

    //std::cout << "my ne " << indicesHost_[NV_] << std::endl;
    GraphElem old_nv[NGPU];
    GraphElem old_ne[NGPU];

    for(int i = 0; i < NGPU; ++i)
    {
        old_nv[i] = nv_[i];
        old_ne[i] = ne_[i];
    }
    determine_edge_device_partition();
    //std::cout << vertexIdsOffsetHost_[NV_-1] << std::endl;
    //nv_per_device_ = (NV_+NGPU-1)/NGPU;
    //std::cout << NV_ << " " << NE_ << std::endl;
    /*for(Int i = 0; i < NV_; ++i)
    {
        GraphElem start = indicesHost_[i+0];
        GraphElem end   = indicesHost_[i+1];
        for(Int j = start; j < end; ++j)
        {
            if(edgesHost_[j] >= NV_)
                std::cout << "error" << std::endl;
        }
    }*/
    //std::cout << NV_ << " " << NE_ << std::endl;
    //for(Int i = 0; i < NV_; ++i)
    //    std::cout << i << " " << indicesHost_[i] << " " << indicesHost_[i+1] << std::endl;
    omp_set_num_threads(NGPU);
    #pragma omp parallel
    {
        int id = omp_get_thread_num() % NGPU;
        CudaSetDevice(id);

        e0_[id] = 0; e1_[id] = 0;
        w0_[id] = 0; w1_[id] = 0;

        GraphElem nv = nv_[id];
        GraphElem ne = ne_[id];
        if(nv > old_nv[id])
        {
            CudaFree(indices_[id]); 
            CudaMalloc(indices_[id], sizeof(GraphElem)*(nv+1));

            CudaFree(vertexWeights_[id]);
            CudaMalloc(vertexWeights_[id], sizeof(GraphWeight)*nv);

            CudaFree(commIds_[id]);
            CudaMalloc(commIds_[id], sizeof(GraphElem)*nv);

            CudaFree(commWeights_[id]);
            CudaMalloc(commWeights_[id], sizeof(GraphWeight)*nv);

            CudaFree(newCommIds_[id]);
            CudaMalloc(newCommIds_[id], sizeof(GraphElem)*nv);

            CudaFree(localCommNums_[id]);
            CudaMalloc(localCommNums_[id], sizeof(GraphElem)*(nv+1));
            CudaMemset(localCommNums_[id], 0, sizeof(GraphElem)*(nv+1));
            local_comm_nums_ptr[id] = thrust::device_pointer_cast(localCommNums_[id]);       
        } 
        if(nv > 0)
        {
            CudaMemcpyAsyncHtoD(indices_[id], indicesHost_+v_base_[id], sizeof(GraphElem)*(nv+1), 0);
            CudaMemset(vertexWeights_[id], 0, sizeof(GraphWeight)*nv);
            CudaMemset(commWeights_[id], 0, sizeof(GraphWeight)*nv);
        }
        CudaMemcpyAsyncHtoD(vertex_per_device_[id], vertex_per_device_host_, sizeof(GraphElem)*(NGPU+1),0); 
        //ne_per_partition_[id] = determine_optimal_edges_per_batch (nv, ne, unit_size);
        //GraphElem ne_per_partition = determine_optimal_edges_per_batch (nv, ne, unit_size);
        GraphElem ne_per_partition = ne_per_partition_[id];

        vertex_partition_[id].clear();
        determine_optimal_vertex_partition(indicesHost_, nv, ne, ne_per_partition, vertex_partition_[id], v_base_[id]);
        sum_vertex_weights(id);
        CudaDeviceSynchronize();

        for(int b = 0; b < nbatches_; ++b)
            vertex_per_batch_partition_[id][b].clear();
    }

    compute_mass();
    maxOrder_ = max_order();
    //if(maxOrder_ > ne_per_partition_cap_)
    //    std::cerr << "WARNING: Max order may exceed buffer size !!!" << std::endl;
    
    //std::cout << "max order is " << maxOrder_ << std::endl;
    for(int i = 0; i < NGPU; ++i)
    {
        if(maxOrder_ > ne_per_partition_[i])
        {
            std::cout <<"Maximum order exceeds the maximum edge capacity\n";
            exit(-1);
        }
    }
    
    for(int i = 0; i < NGPU; ++i)
    {
        CudaSetDevice(i);
        CudaMemcpyHtoD(commIdsPtr_[i],     commIds_,     sizeof(GraphElem*)*NGPU);
        CudaMemcpyHtoD(commWeightsPtr_[i], commWeights_, sizeof(GraphWeight*)*NGPU);
    }

    partition_graph_edge_batch();

    /*std::cout << "total V: " << NV_ << " total E: " << NE_ << std::endl;
    for(int i = 0; i < NGPU; ++i)
    {
        std::cout << nv_[i] << " " << ne_[i] << " " << ne_per_partition_[i] << std::endl;
    }

    for(int i = 0; i <= NGPU; ++i)
        std::cout << i << " " << vertex_per_device_host_[i] << std::endl;
    std::cout << std::endl;
    for(int i = 0; i < NGPU; ++i)
    {
        for(int j = 0; j < nbatches_+1; ++j)
            std::cout << vertex_per_batch_[i][j] << std::endl;
        std::cout << std::endl;
    }
    for(int i = 0; i < NGPU; ++i)
    {
        for(int j = 0; j < nbatches_; ++j)
        {
            std::vector<GraphElem>& vec = vertex_per_batch_partition_[i][j];
            for(int k = 0; k < vec.size(); ++k)
                 std::cout << vertex_per_batch_partition_[i][j][k] << std::endl;
        }
        std::cout << std::endl;
    }*/

    return false;
}

void GraphGPU::dump_partition(const std::string& filename)
{
    clusters_->dump_partition(filename);
}
#endif
