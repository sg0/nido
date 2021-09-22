#include "types.hpp"
#include "cuda_wrapper.hpp"
#include <cooperative_groups.h> 

namespace cg = cooperative_groups;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long  old = *addr_as_ull;
    unsigned long long  assumed;
    do 
    {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,__double_as_longlong(__longlong_as_double(assumed)+val));
    } while(assumed != old);

    return __longlong_as_double(old);
}
#endif

__device__ GraphElem2 search_ranges(GraphElem* ranges, const GraphElem& i)
{
    int a = 0;
    int b = NGPU;
    int c;
    while(a < b)
    {
        c = (a+b)/2;
        if(ranges[c+1] <= i)
            a = c;
        else if(ranges[c] > i)
            b = c;
        else
            break;
    }
    GraphElem start = ranges[c];
    #ifdef USE_32BIT_GRAPH  
    return make_int2(c, i-start);
    #else
    return make_longlong2(c,i-start);
    #endif
}

//#if 0
template<const int WarpSize, const int BlockSize>
__global__
void fill_edges_community_ids_kernel
(
    GraphElem2* __restrict__ commIdKeys,
    GraphElem*  __restrict__ edges,
    GraphElem*  __restrict__ indices,
    GraphElem** __restrict__ commIdsPtr,
    const GraphElem v_base, 
    const GraphElem e_base,
    const GraphElem nv,
    const GraphElem V0,
    GraphElem* vertex_per_device
)
{
    __shared__ GraphElem ranges[BlockSize/WarpSize*2];

    //cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    //const unsigned block_tid = block.thread_rank();
    //const unsigned warp_tid = warp.thread_rank();
    const unsigned warp_tid = threadIdx.x & (WarpSize-1);
    GraphElem* t_ranges = &ranges[(threadIdx.x/WarpSize)*2];

    GraphElem u0 = threadIdx.x/WarpSize+(BlockSize/WarpSize)*blockIdx.x;

    for(GraphElem u = u0; u < nv; u += (BlockSize/WarpSize)*gridDim.x)
    {
        if(warp_tid < 2)
            t_ranges[warp_tid] = indices[u+warp_tid+v_base-V0];
        warp.sync();
 
        GraphElem start = t_ranges[0]-e_base+warp_tid;               
        GraphElem end   = t_ranges[1]-e_base;

        for(GraphElem i = start; i < end; i += WarpSize)
        {
            GraphElem v = edges[i];
            GraphElem2 v_id = search_ranges(vertex_per_device, v);
            GraphElem commId = commIdsPtr[v_id.x][v_id.y]; 
            #ifdef USE_32BIT_GRAPH
            commIdKeys[i] = make_int2(u, commId);
            #else
            commIdKeys[i] = make_longlong2(u, commId);
            #endif
        }
        warp.sync();
    } 
}

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
    GraphElem* vertex_per_device,
    cudaStream_t stream = 0
)
{
    GraphElem nv = v1-v0;
    //std::cout << nv << std::endl;
    long long nblocks = (nv+(BLOCKDIM04/TILESIZE02)-1)/(BLOCKDIM04/TILESIZE02);
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;
    CudaLaunch((fill_edges_community_ids_kernel<TILESIZE02, BLOCKDIM04><<<nblocks, BLOCKDIM04, 0, stream>>>
    (commIdKeys, edges, indices, commIdsPtr, v0, e0, nv, V0, vertex_per_device)));
}

template<const int WarpSize, const int BlockSize>
__global__
void fill_index_orders_kernel
(
    GraphElem* __restrict__ indexOrders,
    GraphElem* __restrict__ indices,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv,
    const GraphElem V0
)
{
    __shared__ GraphElem ranges[(BlockSize/WarpSize)*2];

    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    const unsigned warp_tid = threadIdx.x & (WarpSize-1);

    GraphElem* t_ranges = &ranges[(threadIdx.x/WarpSize)*2];

    GraphElem u0 = (threadIdx.x/WarpSize)+(BlockSize/WarpSize)*blockIdx.x;

    for(GraphElem u = u0; u < nv; u += (BlockSize/WarpSize)*gridDim.x)
    {
        if(warp_tid < 2)
            t_ranges[warp_tid] = indices[u+warp_tid+v_base-V0];
        warp.sync();
 
        GraphElem start = t_ranges[0]-e_base;               
        GraphElem end   = t_ranges[1]-e_base;
        for(GraphElem i = start+warp_tid; i < end; i += WarpSize)
             indexOrders[i] = i-start;
        warp.sync();
    } 
}

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
)
{
    GraphElem nv = v1-v0;
    long long nblocks = (nv+(BLOCKDIM02/TILESIZE01)-1)/(BLOCKDIM02/TILESIZE01); 
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;    
    CudaLaunch((fill_index_orders_kernel<TILESIZE01, BLOCKDIM02><<<nblocks, BLOCKDIM02, 0, stream>>>
    (indexOrders, indices, v0, e0, nv, V0)));
}
//#endif
#if 0
template<const int WarpSize, const int BlockSize>
__global__
void fill_edges_community_ids_kernel
(
    GraphElem2* __restrict__ commIdKeys,
    GraphElem*  __restrict__ edges,
    GraphElem*  __restrict__ indices,
    GraphElem**  __restrict__ commIds,
    const GraphElem v_base, 
    const GraphElem e_base,
    const GraphElem nv
)
{
    __shared__ GraphElem ranges[BlockSize/WarpSize*2];

    //cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    //const unsigned block_tid = block.thread_rank();
    //const unsigned warp_tid = warp.thread_rank();
    const unsigned warp_tid = threadIdx.x & (WarpSize-1);
    GraphElem* t_ranges = &ranges[(threadIdx.x/WarpSize)*2];

    GraphElem u0 = threadIdx.x/WarpSize+(BlockSize/WarpSize)*blockIdx.x;

    for(GraphElem u = u0; u < nv; u += (BlockSize/WarpSize)*gridDim.x)
    {
        if(warp_tid < 2)
            t_ranges[warp_tid] = indices[u+warp_tid+v_base];
        warp.sync();
 
        GraphElem start = t_ranges[0]-e_base+warp_tid;               
        GraphElem end   = t_ranges[1]-e_base;

        for(GraphElem i = start; i < end; i += WarpSize)
        {
            GraphElem commId = commIds[0][edges[i]];    
            #ifdef USE_32BIT_GRAPH
            commIdKeys[i] = make_int2(u, commId);
            #else
            commIdKeys[i] = make_longlong2(u, commId);
            #endif
        }
        warp.sync();
    } 
}

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
)
{
    GraphElem nv = v1-v0;
    //std::cout << nv << std::endl;
    long long nblocks = (nv+(BLOCKDIM02/TILESIZE01)-1)/(BLOCKDIM02/TILESIZE01);
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;
    CudaLaunch((fill_edges_community_ids_kernel<TILESIZE01, BLOCKDIM02><<<nblocks, BLOCKDIM02, 0, stream>>>
    (commIdKeys, edges, indices, commIdsPtr, v0, e0, nv)));
}

template<const int WarpSize, const int BlockSize>
__global__
void fill_index_orders_kernel
(
    GraphElem* __restrict__ indexOrders,
    GraphElem* __restrict__ indices,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv
)
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    const unsigned warp_tid = threadIdx.x & (WarpSize-1);

    GraphElem* t_ranges = &ranges[(threadIdx.x/WarpSize)*2];

    GraphElem u0 = (threadIdx.x/WarpSize)+(BlockSize/WarpSize)*blockIdx.x;

    for(GraphElem u = u0; u < nv; u += (BlockSize/WarpSize)*gridDim.x)
    {
        if(warp_tid < 2)
            t_ranges[warp_tid] = indices[u+warp_tid+v_base];
        warp.sync();
 
        GraphElem start = t_ranges[0]-e_base;               
        GraphElem end   = t_ranges[1]-e_base;
        for(GraphElem i = start+warp_tid; i < end; i += WarpSize)
             indexOrders[i] = i-start;
        warp.sync();
    } 
}

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
)
{
    GraphElem nv = v1-v0;
    long long nblocks = (nv+(BLOCKDIM02/TILESIZE01)-1)/(BLOCKDIM02/TILESIZE01); 
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;    
    CudaLaunch((fill_index_orders_kernel<TILESIZE01, BLOCKDIM02><<<nblocks, BLOCKDIM02, 0, stream>>>
    (indexOrders, indices, v0, e0, nv)));
}
#endif


template<const int WarpSize, const int BlockSize>
__global__
void sum_vertex_weights_kernel
(
    GraphWeight* __restrict__ vertex_weights, 
    GraphWeight* __restrict__ weights,
    GraphElem*   __restrict__ indices,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv,
    const GraphElem V0
)
{   
    __shared__ GraphElem ranges[(BlockSize/WarpSize)*2];

    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    //const unsigned warp_tid = warp.thread_rank();
    const unsigned warp_tid = threadIdx.x & (WarpSize-1);

    GraphElem* t_ranges = &ranges[(threadIdx.x/WarpSize)*2];

    GraphElem step = gridDim.x*BlockSize/WarpSize;
    GraphElem u0 = (threadIdx.x/WarpSize)+BlockSize/WarpSize*blockIdx.x+v_base;
    for(GraphElem u = u0; u < nv+v_base; u += step)
    {   
        if(warp_tid < 2)
            t_ranges[warp_tid] = indices[u+warp_tid-V0];
        warp.sync();
       
        GraphElem start = t_ranges[0]-e_base;
        GraphElem   end = t_ranges[1]-e_base;
        GraphWeight w = 0.; 
        for(GraphElem e = start+warp_tid; e < end; e += WarpSize)
            w += weights[e];

        //warp.sync();
        for(int i = warp.size()/2; i > 0; i/=2)
            w += warp.shfl_down(w, i);
 
        if(warp_tid == 0) 
            vertex_weights[u-V0] = w;
        warp.sync();
    }
}

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
)
{
     GraphElem nv = v1-v0;

     GraphElem nblocks = (nv+(BLOCKDIM01/TILESIZE01)-1)/(BLOCKDIM01/TILESIZE01);
     nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

     CudaLaunch((sum_vertex_weights_kernel<TILESIZE01,BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>
     (vertex_weights, weights, indices, v0, e0, nv, V0)));
}

template<const int BlockSize>
__global__
void compute_community_weights_kernel
(
    GraphWeight** __restrict__ commWeights,
    GraphElem*    __restrict__ commIds,
    GraphWeight*  __restrict__ vertexWeights,
    const GraphElem nv,
    const GraphElem V0,
    GraphElem* vertex_per_device
)
{
    GraphElem u0 = threadIdx.x+BlockSize*blockIdx.x;
    for(GraphElem i = V0+u0; i < V0+nv; i += BlockSize*gridDim.x)
    {
        GraphElem comm_id = commIds[i-V0];
        GraphWeight w = vertexWeights[i-V0];
        GraphElem2 id = search_ranges(vertex_per_device, comm_id);
        GraphWeight* ptr = commWeights[id.x];
        atomicAdd(&ptr[id.y], w);
    }
}

void compute_community_weights_cuda
(
    GraphWeight** commWeights,
    GraphElem*    commIds, 
    GraphWeight* vertexWeights,
    const GraphElem& nv,
    const GraphElem& V0,
    GraphElem* vertex_per_device,
    cudaStream_t stream = 0
)
{
    GraphElem nblocks = (nv+BLOCKDIM01-1) / BLOCKDIM01;
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;
    CudaLaunch((compute_community_weights_kernel<BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>
    (commWeights, commIds, vertexWeights, nv, V0, vertex_per_device)));
}

template<const int BlockSize>
__global__ 
void singleton_partition_kernel
(
    GraphElem*   __restrict__ commIds,
    GraphElem*   __restrict__ newCommIds,
    GraphWeight* __restrict__ commWeights,
    GraphWeight*  __restrict__ vertexWeights,
    const GraphElem nv,
    const GraphElem V0
)
{
    GraphElem u0 = threadIdx.x + BlockSize*blockIdx.x;
    for(GraphElem i = u0+V0; i < nv+V0; i += BlockSize*gridDim.x)
    {
        commIds[i-V0] = i;
        newCommIds[i-V0] = i;
        commWeights[i-V0] = vertexWeights[i-V0];
    }    
}

void singleton_partition_cuda
(
    GraphElem* commIds, 
    GraphElem* newCommIds,
    GraphWeight* commWeights, 
    GraphWeight* vertexWeights, 
    const GraphElem& nv,
    const GraphElem& V0,
    cudaStream_t stream = 0
)
{
    GraphElem nblocks = (nv+BLOCKDIM03-1)/BLOCKDIM03; 
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;
    CudaLaunch((singleton_partition_kernel<BLOCKDIM03><<<nblocks, BLOCKDIM03, 0, stream>>>
    (commIds, newCommIds, commWeights, vertexWeights, nv, V0)));
}

//#if 0
template<const int BlockSize>
__global__
void max_order_reduce_kernel
(
    GraphElem* __restrict__ orders,
    GraphElem* __restrict__ indices, 
    GraphElem nv
)
{
    __shared__ GraphElem max_shared[BlockSize];

    max_shared[threadIdx.x] = 0;

    GraphElem u0 = threadIdx.x + BlockSize * blockIdx.x; 

    for(GraphElem u = u0; u < nv; u += BlockSize*gridDim.x)
    {    
        GraphElem order = indices[u+1]-indices[u];
        if(max_shared[threadIdx.x] < order) 
            max_shared[threadIdx.x] = order;
    }
    __syncthreads();

    for (unsigned int s = BlockSize/2; s >= 32; s>>=1)
    {
        if (threadIdx.x < s && max_shared[threadIdx.x+s] > max_shared[threadIdx.x])
            max_shared[threadIdx.x] = max_shared[threadIdx.x+s];
        __syncthreads();
    }

    GraphElem max = max_shared[threadIdx.x%32];
    for (int offset = 16; offset > 0; offset /= 2)
    {
        GraphElem tmp = __shfl_down_sync(0xffffffff, max, offset);
        max = (tmp > max) ? tmp : max;
        //__syncthreads(); 
    }

    if(threadIdx.x == 0)
        orders[blockIdx.x] = max; 
}

//#endif
template<const int BlockSize>
__global__ 
void max_order_kernel
(
    GraphElem* __restrict__ orders, 
    GraphElem nv
)
{
    __shared__ GraphElem max_shared[BlockSize];

    max_shared[threadIdx.x] = 0;

    GraphElem u0 = threadIdx.x;

    for(GraphElem u = u0; u < nv; u += BlockSize)
    {
        GraphElem order = orders[u];
        if(max_shared[threadIdx.x] < order) 
            max_shared[threadIdx.x] = order;
    }
    __syncthreads();

    for (unsigned int s = BlockSize/2; s >= 32; s>>=1)
    {
        if (threadIdx.x < s && max_shared[threadIdx.x+s] > max_shared[threadIdx.x])
            max_shared[threadIdx.x] = max_shared[threadIdx.x+s];
        __syncthreads();
    }

    GraphElem max = max_shared[threadIdx.x%32];
    for (int offset = 16; offset > 0; offset /= 2)
    {
        GraphElem tmp = __shfl_down_sync(0xffffffff, max, offset);
        max = (tmp > max) ? tmp : max;
        //__syncthreads();
    }

    if(threadIdx.x == 0)
        orders[0] = max;
}

GraphElem max_order_cuda
(
    GraphElem* indices,
    const GraphElem& nv, 
    cudaStream_t stream = 0  
)
{
    GraphElem* max_reduced;

    long long nblocks = (nv + BLOCKDIM01 - 1)/BLOCKDIM01;
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

    CudaMalloc(max_reduced, sizeof(GraphElem)*nblocks);

    CudaLaunch((max_order_reduce_kernel<BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>(max_reduced, indices, nv)));
    CudaLaunch((max_order_kernel<BLOCKDIM01><<<1,BLOCKDIM01, 0, stream>>>(max_reduced, nblocks)));

    GraphElem max;
    CudaMemcpyAsyncDtoH(&max, max_reduced, sizeof(GraphElem), 0);

    CudaFree(max_reduced);

    return max;
}

template<typename T, const int BlockSize>
__global__
void copy_vector_kernel
(
    T* __restrict__ dest, 
    T* __restrict__ src, 
    const GraphElem n
)
{
    const int i0 = threadIdx.x + BlockSize*blockIdx.x;
    for(GraphElem i = i0; i < n; i += BlockSize*gridDim.x)
        dest[i] = src[i]; 
}

void copy_vector_cuda
(
    GraphElem* dest,
    GraphElem* src,
    const GraphElem& ne,
    cudaStream_t stream = 0
)
{
    long long nblocks = (ne + BLOCKDIM03-1)/BLOCKDIM03;
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;
    CudaLaunch((copy_vector_kernel<GraphElem, BLOCKDIM03><<<nblocks, BLOCKDIM03, 0, stream>>>(dest, src, ne)));
}
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
)
{
    GraphElem ne = e1-e0;
    long long nblocks = (ne + BLOCKDIM03-1)/BLOCKDIM03;
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

    CudaLaunch((copy_vector_kernel<GraphElem, BLOCKDIM03><<<nblocks, BLOCKDIM03, 0, stream>>>(dest, src, ne)));
}
*/
//#if 0
template<const int WarpSize, const int BlockSize>
__global__
void reorder_edges_by_keys_kernel
(
    GraphElem* __restrict__ edges,
    GraphElem* __restrict__ indexOrders,
    GraphElem* __restrict__ indices,
    GraphElem* buff,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv,
    const GraphElem V0
)
{
    __shared__ GraphElem ranges[(BlockSize/WarpSize)*2];

    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    const unsigned lane_id = threadIdx.x & (WarpSize-1);

    GraphElem* t_ranges = &ranges[(threadIdx.x/WarpSize)*2];

    GraphElem u0 = (threadIdx.x/WarpSize)+(BlockSize/WarpSize)*blockIdx.x+v_base;
    for(GraphElem u = u0; u < v_base+nv; u += (BlockSize/WarpSize)*gridDim.x)
    {
        if(lane_id < 2)
            t_ranges[lane_id] = indices[u+lane_id-V0];
        warp.sync();

        GraphElem start = t_ranges[0]-e_base; 
        GraphElem end   = t_ranges[1]-e_base;
        for(GraphElem i = start+lane_id; i < end; i += WarpSize)
        {
            GraphElem pos = indexOrders[i];
            buff[i] = edges[pos+start];
        }
        warp.sync();
        for(GraphElem i = start+lane_id; i < end; i += WarpSize)
            edges[i] = buff[i];
        warp.sync();
    } 
}

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
)
{
    GraphElem nv = v1-v0;

    long long nblocks = (nv+(BLOCKDIM02/TILESIZE01)-1)/(BLOCKDIM02/TILESIZE01);
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

    CudaLaunch((reorder_edges_by_keys_kernel<TILESIZE01,BLOCKDIM02><<<nblocks, BLOCKDIM02, 0, stream>>>
    (edges, indexOrders, indices, buff, v0, e0, nv, V0)));
}

template<const int WarpSize, const int BlockSize>
__global__
void reorder_weights_by_keys_kernel
(
    GraphWeight* __restrict__ edgeWeights,
    GraphElem*   __restrict__ indexOrders,
    GraphElem*   __restrict__ indices,
    GraphWeight* buff,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv,
    const GraphElem V0
)
{
    __shared__ GraphElem ranges[(BlockSize/WarpSize)*2];

    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    const unsigned lane_id = threadIdx.x & (WarpSize-1);
    GraphElem* t_ranges = &ranges[(threadIdx.x/WarpSize)*2];

    GraphElem u0 = (threadIdx.x/WarpSize)+(BlockSize/WarpSize)*blockIdx.x+v_base;
    for(GraphElem u = u0; u < v_base+nv; u += (BlockSize/WarpSize)*gridDim.x)
    {
        if(lane_id < 2)
            t_ranges[lane_id] = indices[u+lane_id-V0];
        warp.sync();

        GraphElem start = t_ranges[0]-e_base; 
        GraphElem end   = t_ranges[1]-e_base;
        for(GraphElem i = start+lane_id; i < end; i += WarpSize)
        {
            GraphElem pos = indexOrders[i];
            buff[i] = edgeWeights[pos+start];
        }
        warp.sync();
        for(GraphElem i = start+lane_id; i < end; i += WarpSize)
            edgeWeights[i] = buff[i];
        warp.sync();
    } 
}

void reorder_weights_by_keys_cuda
( 
    GraphWeight* edgeWeights, 
    GraphElem*   indexOrders, 
    GraphElem*   indices , 
    GraphWeight* buff, 
    const GraphElem& v0, 
    const GraphElem& v1,  
    const GraphElem& e0, 
    const GraphElem& e1,
    const GraphElem& V0,
    cudaStream_t stream = 0
)
{
    GraphElem nv = v1-v0; 
    long long nblocks = (nv+(BLOCKDIM02/TILESIZE01)-1)/(BLOCKDIM02/TILESIZE01);
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

    CudaLaunch((reorder_weights_by_keys_kernel<TILESIZE01,BLOCKDIM02><<<nblocks, BLOCKDIM02,0,stream>>>
    (edgeWeights, indexOrders, indices, buff, v0, e0, nv, V0)));
}
//#endif

#if 0
template<const int WarpSize, const int BlockSize>
__global__
void reorder_edges_by_keys_kernel
(
    GraphElem* __restrict__ edges,
    GraphElem* __restrict__ indexOrders,
    GraphElem* __restrict__ indices,
    GraphElem* buff,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv
)
{
    __shared__ GraphElem ranges[(BlockSize/WarpSize)*2];

    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    const unsigned lane_id = threadIdx.x & (WarpSize-1);

    GraphElem* t_ranges = &ranges[(threadIdx.x/WarpSize)*2];

    GraphElem u0 = (threadIdx.x/WarpSize)+(BlockSize/WarpSize)*blockIdx.x+v_base;
    for(GraphElem u = u0; u < v_base+nv; u += (BlockSize/WarpSize)*gridDim.x)
    {
        if(lane_id < 2)
            t_ranges[lane_id] = indices[u+lane_id];
        warp.sync();

        GraphElem start = t_ranges[0]-e_base; 
        GraphElem end   = t_ranges[1]-e_base;
        for(GraphElem i = start+lane_id; i < end; i += WarpSize)
        {
            GraphElem pos = indexOrders[i];
            buff[i] = edges[pos+start];
        }
        warp.sync();
        for(GraphElem i = start+lane_id; i < end; i += WarpSize)
            edges[i] = buff[i];
        warp.sync();
    } 
}

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
)
{
    GraphElem nv = v1-v0;

    long long nblocks = (nv+(BLOCKDIM02/TILESIZE01)-1)/(BLOCKDIM02/TILESIZE01);
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

    CudaLaunch((reorder_edges_by_keys_kernel<TILESIZE01,BLOCKDIM02><<<nblocks, BLOCKDIM02, 0, stream>>>
    (edges, indexOrders, indices, buff, v0, e0, nv)));
}

template<const int WarpSize, const int BlockSize>
__global__
void reorder_weights_by_keys_kernel
(
    GraphWeight* __restrict__ edgeWeights,
    GraphElem*   __restrict__ indexOrders,
    GraphElem*   __restrict__ indices,
    GraphWeight* buff,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv
)
{
    __shared__ GraphElem ranges[(BlockSize/WarpSize)*2];

    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    const unsigned lane_id = threadIdx.x & (WarpSize-1);
    GraphElem* t_ranges = &ranges[(threadIdx.x/WarpSize)*2];

    GraphElem u0 = (threadIdx.x/WarpSize)+(BlockSize/WarpSize)*blockIdx.x+v_base;
    for(GraphElem u = u0; u < v_base+nv; u += (BlockSize/WarpSize)*gridDim.x)
    {
        if(lane_id < 2)
            t_ranges[lane_id] = indices[u+lane_id];
        warp.sync();

        GraphElem start = t_ranges[0]-e_base; 
        GraphElem end   = t_ranges[1]-e_base;
        for(GraphElem i = start+lane_id; i < end; i += WarpSize)
        {
            GraphElem pos = indexOrders[i];
            buff[i] = edgeWeights[pos+start];
        }
        warp.sync();
        for(GraphElem i = start+lane_id; i < end; i += WarpSize)
            edgeWeights[i] = buff[i];
        warp.sync();
    } 
}

void reorder_weights_by_keys_cuda
(
    GraphWeight* edgeWeights, 
    GraphElem*   indexOrders, 
    GraphElem*   indices , 
    GraphWeight* buff, 
    const GraphElem& v0, 
    const GraphElem& v1,  
    const GraphElem& e0, 
    const GraphElem& e1,
    const GraphElem& V0,
    cudaStream_t stream = 0
)
{
    GraphElem nv = v1-v0; 
    long long nblocks = (nv+(BLOCKDIM02/TILESIZE01)-1)/(BLOCKDIM02/TILESIZE01);
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

    CudaLaunch((reorder_weights_by_keys_kernel<TILESIZE01,BLOCKDIM02><<<nblocks, BLOCKDIM02,0,stream>>>
    (edgeWeights, indexOrders, indices, buff, v0, e0, nv)));
}
#endif

#if 0
template<const int BlockSize, const int WarpSize=32>
__global__
void build_local_commid_offsets_kernel
(
    GraphElem* localOffsets,
    GraphElem* localCommNums,
    GraphElem*   __restrict__ edges,
    GraphElem*   __restrict__ indices,
    GraphElem**  __restrict__ commIdsPtr,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv,
    const GraphElem V0,
    const GraphElem nv_per_device
)
{  
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());
    unsigned lane_id = threadIdx.x &(WarpSize-1);
    GraphElem v1 = (nv + (BlockSize/WarpSize)*gridDim.x-1)/((BlockSize/WarpSize)*gridDim.x);
    GraphElem v0 = (threadIdx.x / WarpSize + (BlockSize/WarpSize)*blockIdx.x)*v1;
    v1 += v0;
    if(v0 > nv) v0 = nv;
    if(v1 > nv) v1 = nv;

    v0 += v_base;
    v1 += v_base;

    GraphElem start = indices[v0-V0]-e_base;
    GraphElem end = 0;
    for(GraphElem v = v0; v < v1; ++v)
    {
        if(lane_id == 0)
            end   = indices[v+1-V0]-e_base;
        end   = warp.shfl(end, 0);
        
        volatile GraphElem count = 0;
        volatile GraphElem target;
        volatile GraphElem localId = 0;

        while(count < end-start)
        {
            if(lane_id == 0x00)
                localOffsets[start+localId] = count;
            GraphElem f = edges[start+count]; 
            target = commIdsPtr[f/nv_per_device][f%nv_per_device];
            volatile unsigned localCount = 0;
            for(GraphElem u = start+count; u < end; u += WarpSize)
            {
                if((u+lane_id) < end)
                {
                    GraphElem g = edges[u+lane_id];
                    if(commIdsPtr[g/nv_per_device][g%nv_per_device] == target)
                        localCount++;
                    else
                        break;
                }
            }
            ///warp.sync();
            #pragma unroll
            for(int i = WarpSize/2; i > 0; i/=2)
                localCount += warp.shfl_down(localCount, i);
            count += localCount;
            count = warp.shfl(count, 0);
            localId++;
        }
        start = end;
        if(lane_id == 0x00)
            localCommNums[v-v_base] = localId;
        ///warp.sync();
    }
}
#endif
#if 0
template<const int BlockSize, const int WarpSize=32>
__global__
void build_local_commid_offsets_kernel
(
    GraphElem* localOffsets,
    GraphElem* localCommNums,
    GraphElem*   __restrict__ edges,
    GraphElem*   __restrict__ indices,
    GraphElem**   __restrict__ commIds,
    GraphElem v_base,
    GraphElem e_base,
    GraphElem nv
)
{  
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());
    unsigned lane_id = threadIdx.x &(WarpSize-1);
    GraphElem v1 = (nv + (BlockSize/WarpSize)*gridDim.x-1)/((BlockSize/WarpSize)*gridDim.x);
    GraphElem v0 = (threadIdx.x / WarpSize + (BlockSize/WarpSize)*blockIdx.x)*v1;
    v1 += v0;
    if(v1 > nv) v1 = nv;

    v0 += v_base;
    v1 += v_base;

    GraphElem start = indices[v0]-e_base;
    GraphElem end = 0;
    for(GraphElem v = v0; v < v1; ++v)
    {
        if(lane_id == 0)
            end   = indices[v+1]-e_base;
        end   = warp.shfl(end, 0);
        
        volatile GraphElem count = 0;
        volatile GraphElem target;
        volatile GraphElem localId = 0;

        while(count < end-start)
        {
            if(lane_id == 0x00)
                localOffsets[start+localId] = count; 
            target = commIds[0][edges[start+count]];
            volatile unsigned localCount = 0;
            for(GraphElem u = start+count; u < end; u += WarpSize)
            {
                if((u+lane_id) < end)
                {
                    if(commIds[0][edges[u+lane_id]] == target)
                        localCount++;
                    else
                        break;
                }
            }
            ///warp.sync();
            #pragma unroll
            for(int i = WarpSize/2; i > 0; i/=2)
                localCount += warp.shfl_down(localCount, i);
            count += localCount;
            count = warp.shfl(count, 0);
            localId++;
        }
        start = end;
        if(lane_id == 0x00)
            localCommNums[v-v_base] = localId;
        ///warp.sync();
    }
}
#endif

template<const int BlockSize, const int WarpSize=32>
__global__
void build_local_commid_offsets_kernel
(
    GraphElem*  __restrict__ localOffsets,
    GraphElem*  __restrict__ localCommNums,
    GraphElem2* __restrict__ commIdKeys,
    //GraphElem*   __restrict__ edges,
    GraphElem*   __restrict__ indices,
    //GraphElem**  __restrict__ commIdsPtr,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv,
    const GraphElem V0
)
{  
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());
    unsigned lane_id = threadIdx.x &(WarpSize-1);
    GraphElem v1 = (nv + (BlockSize/WarpSize)*gridDim.x-1)/((BlockSize/WarpSize)*gridDim.x);
    GraphElem v0 = (threadIdx.x / WarpSize + (BlockSize/WarpSize)*blockIdx.x)*v1;
    v1 += v0;
    if(v0 > nv) v0 = nv;
    if(v1 > nv) v1 = nv;

    v0 += v_base;
    v1 += v_base;

    GraphElem start = indices[v0-V0]-e_base;
    GraphElem end = 0;
    for(GraphElem v = v0; v < v1; ++v)
    {
        if(lane_id == 0)
            end   = indices[v+1-V0]-e_base;
        end   = warp.shfl(end, 0);
        
        volatile GraphElem  count = 0;
        GraphElem2 target;
        volatile GraphElem  localId = 0;

        while(count < end-start)
        {
            if(lane_id == 0x00)
                localOffsets[start+localId] = count;
            //GraphElem f = edges[start+count];
            target = commIdKeys[start+count]; 
            //target = commIdsPtr[f/nv_per_device][f%nv_per_device];
            //target = f.y;
            volatile unsigned localCount = 0;
            for(GraphElem u = start+count; u < end; u += WarpSize)
            {
                if((u+lane_id) < end)
                {
                    //GraphElem g = edges[u+lane_id];
                    GraphElem2 g = commIdKeys[u+lane_id];
                    //if(commIdsPtr[g/nv_per_device][g%nv_per_device] == target)
                    if(g.y == target.y)
                        localCount++;
                    else
                        break;
                }
            }
            //warp.sync();
            #pragma unroll
            for(int i = WarpSize/2; i > 0; i/=2)
                localCount += warp.shfl_down(localCount, i);
            count += localCount;
            count = warp.shfl(count, 0);
            localId++;
        }
        start = end;
        if(lane_id == 0x00)
            localCommNums[v-v_base] = localId;
        //warp.sync();
    }
}

void build_local_commid_offsets_cuda
(
    GraphElem*  localOffsets,
    GraphElem*  localCommNums,
    GraphElem2* commIdKeys,
    //GraphElem*  edges,
    GraphElem*  indices,
    //GraphElem** commIdsPtr,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    const GraphElem& V0,
    cudaStream_t stream = 0
)
{
    GraphElem nv = v1-v0;
    long long nblocks = (nv+(BLOCKDIM04/8-1))/(BLOCKDIM04/8);
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

    CudaLaunch((build_local_commid_offsets_kernel<BLOCKDIM04,8><<<nblocks, BLOCKDIM04, 0, stream>>>
    (localOffsets, localCommNums, commIdKeys, indices, v0, e0, nv, V0)));
    //CudaLaunch((build_local_commid_offsets_kernel<BLOCKDIM04,2><<<nblocks, BLOCKDIM04, 0, stream>>>
    //(localOffsets, localCommNums, edges, indices, commIdsPtr, v0, e0, nv, V0, nv_per_device)));
    //CudaLaunch((build_local_commid_offsets_kernel<BLOCKDIM03,16><<<nblocks, BLOCKDIM03, 0, stream>>>
    //(localOffsets,localCommNums, edges, indices, commIdsPtr, v0, e0, nv)));
}
/*
template<const int BlockSize>
__global__
void update_commids_kernel
(
    GraphElem*   __restrict__ commIds,
    GraphElem*   __restrict__ newCommIds,
    GraphWeight* __restrict__ commWeights,
    GraphWeight* __restrict__ vertexWeights,
    const GraphElem v0,
    const GraphElem v1
)
{
    for(GraphElem v = threadIdx.x + BlockSize*blockIdx.x+v0; v < v1; v += BlockSize*gridDim.x)
    {
        GraphElem src = commIds[v];
        GraphElem dest = newCommIds[v];
        if(src != dest)
        {
            //GraphWeight ki = vertexWeights[v];
            //atomicAdd(commWeights+src, -ki);
            //atomicAdd(commWeights+dest, ki);
            commIds[v] = dest;
            newCommIds[v] = src;        
        } 
    }
}

void update_commids_cuda
(
    GraphElem* commIds,
    GraphElem* newCommIds,
    GraphWeight* commWeights,    
    GraphWeight* vertexWeights,
    const GraphElem& v0,
    const GraphElem& v1,
    cudaStream_t stream = 0
)
{
    long long nblocks = (v1-v0+BLOCKDIM02-1)/BLOCKDIM02;
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

    CudaLaunch((update_commids_kernel<BLOCKDIM02><<<nblocks, BLOCKDIM02, 0, stream>>>
    (commIds,newCommIds,commWeights, vertexWeights, v0, v1)));
}
*/

template<typename T, const int BlockSize>
__global__
void exchange_vector_kernel
(
    T* __restrict__ dest,
    T* __restrict__ src,
    const GraphElem nv
)
{
    for(GraphElem i = threadIdx.x+BlockSize*blockIdx.x; i < nv; i += BlockSize*gridDim.x)
    {
        T val = dest[i];
        dest[i] = src[i];
        src[i] = val;
    }
}

template<typename T>
void exchange_vector_cuda
(
    T* dest,
    T* src,
    const GraphElem& nv,
    cudaStream_t stream = 0
)
{
    long long nblocks = (nv+BLOCKDIM02-1)/BLOCKDIM02;
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks; 
  
    CudaLaunch((exchange_vector_kernel<T, BLOCKDIM02><<<nblocks, BLOCKDIM02, 0, stream>>>(dest, src, nv)));
}

template void exchange_vector_cuda<GraphElem>
(
    GraphElem* dest,
    GraphElem* src,
    const GraphElem& nv,
    cudaStream_t stream
);
 
template<const int BlockSize>
__global__
void update_community_weights_kernel
(
    GraphWeight** __restrict__ commWeightsPtr,
    GraphElem*    __restrict__ commIds,
    GraphElem*    __restrict__ newCommIds,
    GraphWeight*  __restrict__ vertexWeights,
    const GraphElem nv,
    GraphElem* vertex_per_device
)
{
    for(GraphElem v = threadIdx.x + BlockSize*blockIdx.x; v < nv; v += BlockSize*gridDim.x)
    {
        GraphElem src = commIds[v];
        GraphElem dest = newCommIds[v];
        if(src != dest)
        {
            GraphWeight ki = vertexWeights[v];

            GraphElem2 id = search_ranges(vertex_per_device, src);
            GraphWeight* ptr = commWeightsPtr[id.x];
            atomicAdd(&ptr[id.y], -ki);

            id = search_ranges(vertex_per_device, dest);
            ptr = commWeightsPtr[id.x]; 
            atomicAdd(&ptr[id.y], ki);
        }
    }
}

void update_community_weights_cuda
(
    GraphWeight** commWeightsPtr, 
    GraphElem* commIds, 
    GraphElem* newCommIds, 
    GraphWeight* vertexWeights, 
    const GraphElem& nv,
    GraphElem* vertex_per_device,
    cudaStream_t stream = 0
)
{
    long long nblocks = (nv+BLOCKDIM01-1)/BLOCKDIM01;
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

    CudaLaunch((update_community_weights_kernel<BLOCKDIM01>
    <<<nblocks, BLOCKDIM01, 0, stream>>>(commWeightsPtr, commIds, newCommIds, vertexWeights, nv, vertex_per_device)));
}
//#if 0
template<const int BlockSize, const int WarpSize, const int TileSize>
__global__
void louvain_update_kernel
(
    GraphElem*    __restrict__ localCommOffsets,
    GraphElem*    __restrict__ localCommNums,
    GraphElem*    __restrict__ edges,
    GraphWeight*  __restrict__ edgeWeights,
    GraphElem*    __restrict__ indices,
    GraphWeight*  __restrict__ vertexWeights,
    GraphElem*    __restrict__ commIds,
    GraphElem**   __restrict__ commIdsPtr,
    GraphWeight** __restrict__ commWeightsPtr,
    GraphElem*   __restrict__ newCommIds,
    const GraphWeight mass,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv,
    const GraphElem V0,
    GraphElem* vertex_per_device
)
{
    __shared__ GraphWeight self_shared[BlockSize/WarpSize];
    __shared__ Float2 gain_shared[BlockSize/TileSize];

    GraphWeight gain, selfWeight;
    Float2 target;

    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());
    cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(warp);

    const unsigned warp_id      = threadIdx.x / WarpSize;
    const unsigned lane_id      = threadIdx.x & (WarpSize-1);
    const unsigned tile_id      = lane_id / TileSize;
    const unsigned tile_lane_id = threadIdx.x & (TileSize-1);

    GraphElem v1 = (nv + (BlockSize/WarpSize)*gridDim.x-1)/((BlockSize/WarpSize)*gridDim.x);
    GraphElem v0 = (warp_id + (BlockSize/WarpSize)*blockIdx.x)*v1;

    v1 += v0;
    if(v0 > nv) v0 = nv;
    if(v1 > nv) v1 = nv;
    v0 += v_base;
    v1 += v_base;
    GraphElem start, end;
    start = indices[v0-V0]-e_base;
    
    for(GraphElem v = v0; v < v1; ++v)
    {
        selfWeight = 0;
        #ifdef USE_32BIT_GRAPH
        target = make_float2(-MAX_FLOAT, __int_as_float(0));
        #else
        target = make_double2(-MAX_FLOAT, __longlong_as_double(0LL));
        #endif
        //warp.sync();
        if(lane_id == 0x00)
            self_shared[warp_id] = 0;
        //warp.sync();
        if(lane_id == 0x00)
            end = indices[v+1-V0]-e_base;
        end = warp.shfl(end, 0);
        
        GraphElem localCommNum = localCommNums[v-v_base];
        GraphElem myCommId = commIds[v-V0];
        GraphWeight ki = vertexWeights[v-V0];

        //loop throught unique community ids
        for(GraphElem j = tile_id; j < localCommNum; j += WarpSize/TileSize)
        {
            GraphElem n0, n1;
            n0 = localCommOffsets[j+start]+start;
            n1 = ((j == localCommNum-1) ? end : localCommOffsets[j+start+1]+start);
            gain = 0.;
            GraphElem g = edges[n0];
            GraphElem2 g_id = search_ranges(vertex_per_device, g);
            GraphElem destCommId = commIdsPtr[g_id.x][g_id.y];
            if(destCommId == myCommId)
            {
                for(GraphElem k = n0+tile_lane_id; k < n1; k+=TileSize)
                {
                    if(edges[k] != v)
                        selfWeight += edgeWeights[k];
                }
                for(unsigned int i = TileSize/2; i > 0; i/=2)
                    selfWeight += tile.shfl_down(selfWeight, i);
                if(tile_lane_id == 0x00)
                    self_shared[warp_id] = selfWeight;
            }
            else
            {          
                for(GraphElem k = n0+tile_lane_id; k < n1; k+=TileSize)
                    gain += edgeWeights[k]; 
                for(unsigned int i = TileSize/2; i > 0; i/=2)
                    gain += tile.shfl_down(gain, i);
                if(tile_lane_id == 0x00)
                {
                    GraphElem2 destCommId_id = search_ranges(vertex_per_device, destCommId);
                    gain -= ki*commWeightsPtr[destCommId_id.x][destCommId_id.y]/(2.*mass);
                    gain /= mass;
                    if(target.x < gain)
                    {
                        target.x = gain;
                        target.y = __longlong_as_double(destCommId);
                    }
                }
            }
            //tile.sync();
        }
        //warp.sync();
        if(tile_lane_id == 0x00)
            gain_shared[(WarpSize/TileSize)*warp_id+tile_id] = target;
        warp.sync();
        
        #pragma unroll
        for(unsigned int i = WarpSize/(TileSize*2); i > 0; i/=2)
        {
            if(lane_id < i)
            {
                if(gain_shared[(WarpSize/TileSize)*warp_id+lane_id+i].x > 
                   gain_shared[(WarpSize/TileSize)*warp_id+lane_id+0].x)
                {
                    gain_shared[(WarpSize/TileSize)*warp_id+lane_id+0] = 
                    gain_shared[(WarpSize/TileSize)*warp_id+lane_id+i];
                }
            }
            warp.sync();
        }
        if(lane_id == 0)
        {
            gain = gain_shared[(WarpSize/TileSize)*warp_id].x;
            localCommNum = __double_as_longlong(gain_shared[(WarpSize/TileSize)*warp_id].y);
            selfWeight = self_shared[warp_id];
            GraphElem2 myCommId_id = search_ranges(vertex_per_device, myCommId); 
            selfWeight -= ki*(commWeightsPtr[myCommId_id.x][myCommId_id.y]-ki)/(2.*mass); 
            selfWeight /= mass;
            gain -= selfWeight;
            if(gain > 0)
                newCommIds[v-V0] = localCommNum;
            else
                newCommIds[v-V0] = myCommId;
        }
        warp.sync();
        start = end;
    }
}

void louvain_update_cuda
(
    GraphElem* localCommOffsets, 
    GraphElem* localCommNums, 
    GraphElem*   edges,
    GraphWeight* edgeWeights,
    GraphElem*   indices, 
    GraphWeight* vertexWeights, 
    GraphElem*    commIds,
    GraphElem**   commIdsPtr, 
    GraphWeight** commWeightsPtr, 
    GraphElem*   newCommIds,
    const GraphWeight& mass, 
    const GraphElem& v0, 
    const GraphElem& v1, 
    const GraphElem& e0, 
    const GraphElem& e1,
    const GraphElem& V0,
    GraphElem* vertex_per_device,
    cudaStream_t stream = 0
)
{
    const GraphElem nv = v1-v0;

    long long nblocks = (nv + (BLOCKDIM03/TILESIZE01-1))/(BLOCKDIM03/TILESIZE01);
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

    CudaLaunch((louvain_update_kernel<BLOCKDIM03,TILESIZE01,TILESIZE02><<<nblocks, BLOCKDIM03, 0, stream>>>
    (localCommOffsets, localCommNums, edges, edgeWeights, indices, vertexWeights, commIds, commIdsPtr, 
     commWeightsPtr, newCommIds, mass, v0, e0, nv, V0, vertex_per_device)));
}
//#endif
#if 0
template<const int BlockSize, const int WarpSize, const int TileSize>
__global__
void louvain_update_kernel
(
    GraphElem* localCommOffsets,
    GraphElem* localCommNums,
    GraphElem*   __restrict__ edges,
    GraphWeight* __restrict__ edgeWeights,
    GraphElem*   __restrict__ indices,
    GraphWeight* __restrict__ vertexWeights,
    GraphElem*   __restrict__ commIds,
    GraphWeight** __restrict__ commWeights,
    GraphElem*   __restrict__ newCommIds,
    const GraphWeight mass,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv
)
{
    __shared__ GraphWeight self_shared[BlockSize/WarpSize];
    __shared__ Float2 gain_shared[BlockSize/TileSize];

    GraphWeight gain, selfWeight;
    Float2 target;

    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());
    cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(warp);

    const unsigned warp_id      = threadIdx.x / WarpSize;
    const unsigned lane_id      = threadIdx.x & (WarpSize-1);
    const unsigned tile_id      = lane_id / TileSize;
    const unsigned tile_lane_id = threadIdx.x & (TileSize-1);

    GraphElem v1 = (nv + (BlockSize/WarpSize)*gridDim.x-1)/((BlockSize/WarpSize)*gridDim.x);
    GraphElem v0 = (warp_id + (BlockSize/WarpSize)*blockIdx.x)*v1;

    v1 += v0;
    if(v1 > nv) v1 = nv;
    v0 += v_base;
    v1 += v_base;
    GraphElem start, end;
    start = indices[v0]-e_base;
    
    for(GraphElem v = v0; v < v1; ++v)
    {
        selfWeight = 0;
        #ifdef USE_32BIT_GRAPH
        target = make_float2(-MAX_FLOAT, __int_as_float(0));
        #else
        target = make_double2(-MAX_FLOAT, __longlong_as_double(0LL));
        #endif

        if(lane_id == 0x00)
            self_shared[warp_id] = 0;

        if(lane_id == 0x00)
            end = indices[v+1]-e_base;
        end = warp.shfl(end, 0);
        
        GraphElem localCommNum = localCommNums[v-v_base];
        GraphElem myCommId = commIds[v];
        GraphWeight ki = vertexWeights[v];

        //loop throught unique community ids
        for(GraphElem j = tile_id; j < localCommNum; j += WarpSize/TileSize)
        {
            GraphElem n0, n1;
            n0 = localCommOffsets[j+start]+start;
            n1 = ((j == localCommNum-1) ? end : localCommOffsets[j+start+1]+start);
            gain = 0.;
            GraphElem destCommId = commIds[edges[n0]];
            if(destCommId == myCommId)
            {
                for(GraphElem k = n0+tile_lane_id; k < n1; k+=TileSize)
                {
                    if(edges[k] != v)
                        selfWeight += edgeWeights[k];
                }
                for(unsigned int i = TileSize/2; i > 0; i/=2)
                    selfWeight += tile.shfl_down(selfWeight, i);
                if(tile_lane_id == 0x00)
                    self_shared[warp_id] = selfWeight;
            }
            else
            {          
                for(GraphElem k = n0+tile_lane_id; k < n1; k+=TileSize)
                    gain += edgeWeights[k];
                for(unsigned int i = TileSize/2; i > 0; i/=2)
                    gain += tile.shfl_down(gain, i);
                if(tile_lane_id == 0x00)
                {
                    gain -= ki*commWeights[0][destCommId]/(2.*mass);
                    gain /= mass;
                    if(target.x < gain)
                    {
                        target.x = gain;
                        target.y = __longlong_as_double(destCommId);
                    }
                }
            }
            //tile.sync();
        }
        //warp.sync();
        if(tile_lane_id == 0x00)
            gain_shared[(WarpSize/TileSize)*warp_id+tile_id] = target;
        warp.sync();
        
        #pragma unroll
        for(unsigned int i = WarpSize/(TileSize*2); i > 0; i/=2)
        {
            if(lane_id < i)
            {
                if(gain_shared[(WarpSize/TileSize)*warp_id+lane_id+i].x > 
                   gain_shared[(WarpSize/TileSize)*warp_id+lane_id+0].x)
                {
                    gain_shared[(WarpSize/TileSize)*warp_id+lane_id+0] = 
                    gain_shared[(WarpSize/TileSize)*warp_id+lane_id+i];
                }
            }
            warp.sync();
        }
        if(lane_id == 0)
        {
            gain = gain_shared[(WarpSize/TileSize)*warp_id].x;
            localCommNum = __double_as_longlong(gain_shared[(WarpSize/TileSize)*warp_id].y);
            selfWeight = self_shared[warp_id];
            selfWeight -= ki*(commWeights[0][myCommId]-ki)/(2.*mass); 
            selfWeight /= mass;
            gain -= selfWeight;
            if(gain > 0)
                newCommIds[v] = localCommNum;
            else
                newCommIds[v] = myCommId;
        }
        warp.sync();
        start = end;
    }
}
//#endif

void louvain_update_cuda
(
    GraphElem* localCommOffsets, 
    GraphElem* localCommNums, 
    GraphElem*   edges,
    GraphWeight* edgeWeights,
    GraphElem*   indices, 
    GraphWeight* vertexWeights, 
    GraphElem*    commIds,
    GraphElem**   commIdsPtr, 
    GraphWeight** commWeightsPtr, 
    GraphElem*   newCommIds,
    const GraphWeight& mass, 
    const GraphElem& v0, 
    const GraphElem& v1, 
    const GraphElem& e0, 
    const GraphElem& e1,
    const GraphElem& V0,
    const GraphElem& nv_per_device,
    cudaStream_t stream = 0
)
{
    const GraphElem nv = v1-v0;

    long long nblocks = (nv + (BLOCKDIM02/TILESIZE01-1))/(BLOCKDIM02/TILESIZE01);
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

    CudaLaunch((louvain_update_kernel<BLOCKDIM02,TILESIZE01,TILESIZE02><<<nblocks, BLOCKDIM02, 0, stream>>>
    (localCommOffsets, localCommNums, edges, edgeWeights, indices, vertexWeights, commIds, commWeightsPtr, 
     newCommIds, mass, v0, e0, v1-v0)));
}
#endif

template<const int WarpSize=32>
__global__
void compute_mass_reduce_kernel
(
    GraphWeight* __restrict__ mass, 
    GraphWeight* __restrict__ vertexWeights, 
    GraphElem nv
)
{ 
    GraphWeight m = 0.;
    for(GraphElem i = threadIdx.x+WarpSize*blockIdx.x; i < nv; i += WarpSize*gridDim.x)
        m += vertexWeights[i];
    //__syncthreads();
    for(unsigned int i = WarpSize/2; i > 0; i/=2)
        m += __shfl_down_sync(0xffffffff, m, i, WarpSize);

    if(threadIdx.x ==0)
        mass[blockIdx.x] = m;
}

template<const int WarpSize=32>
__global__
void reduce_vector_kernel
(
    GraphWeight* __restrict__ mass,
    //GraphWeight* vertexWeights,
    GraphElem nv
)
{
    __shared__ GraphWeight m_shared[WarpSize];
    GraphWeight m = 0.;
    for(GraphElem i = threadIdx.x; i < nv; i += WarpSize*WarpSize)
        m += mass[i];
    //__syncthreads();

    for(unsigned int i = WarpSize/2; i > 0; i/=2)
        m += __shfl_down_sync(0xffffffff, m, i, WarpSize);

    if((threadIdx.x & (WarpSize-1)) == 0)
        m_shared[threadIdx.x/WarpSize] = m;
    __syncthreads();
    if(threadIdx.x / WarpSize == 0)
    {
        m = m_shared[threadIdx.x & (WarpSize-1)];
        for(unsigned int i = WarpSize/2; i > 0; i/=2)
            m += __shfl_down_sync(0xffffffff, m, i, WarpSize);
    }
    if(threadIdx.x ==0)
        *mass = m;
}

GraphWeight compute_mass_cuda
(
    GraphWeight* vertexWeights,
    GraphElem nv,
    cudaStream_t stream = 0
)
{
    GraphWeight *mass;
    const int nblocks = 4096;
    CudaMalloc(mass, sizeof(GraphWeight)*nblocks);

    CudaLaunch((compute_mass_reduce_kernel<WARPSIZE><<<nblocks, WARPSIZE>>>(mass, vertexWeights, nv)));
    CudaLaunch((reduce_vector_kernel<WARPSIZE><<<1, WARPSIZE*WARPSIZE>>>(mass, nblocks)));

    GraphWeight m;
    CudaMemcpyDtoH(&m, mass, sizeof(GraphWeight));
    CudaFree(mass);

    return 0.5*m;
}
//#if 0
template<const int BlockSize, const int WarpSize>
__global__
void compute_modularity_reduce_kernel
(
    GraphWeight*  __restrict__ mod,
    GraphElem*    __restrict__ edges,
    GraphWeight*  __restrict__ edgeWeights,
    GraphElem*    __restrict__ indices,
    GraphElem*    __restrict__ commIds,
    GraphElem**   __restrict__ commIdsPtr,
    GraphWeight** __restrict__ commWeightsPtr,
    GraphElem*    __restrict__ localCommOffsets,
    GraphElem*    __restrict__ localCommNums,
    const GraphWeight mass,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv,
    const GraphElem V0,
    GraphElem* vertex_per_device
)
{
    //__shared__ GraphWeight self_shared[BlockSize/WarpSize];
    GraphWeight selfWeight;

    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    const unsigned warp_id      = threadIdx.x / WarpSize;
    const unsigned lane_id      = threadIdx.x & (WarpSize-1);

    GraphElem v1 = (nv + (BlockSize/WarpSize)*gridDim.x-1)/((BlockSize/WarpSize)*gridDim.x);
    GraphElem v0 = (warp_id + (BlockSize/WarpSize)*blockIdx.x)*v1;

    v1 += v0;
    if(v0 > nv) v0 = nv;
    if(v1 > nv) v1 = nv;
    v0 += v_base;
    v1 += v_base;
    GraphElem start, end;
    start = indices[v0-V0]-e_base;
    
    for(GraphElem v = v0; v < v1; ++v)
    {
        selfWeight = 0;

        if(lane_id == 0x00)
            end = indices[v+1-V0]-e_base;
        end = warp.shfl(end, 0);
        
        GraphElem localCommNum = localCommNums[v-v_base];
        GraphElem myCommId = commIds[v-V0];
        //loop throught unique community ids
        for(GraphElem j = 0; j < localCommNum; ++j)
        {
            GraphElem n0, n1;
            n0 = localCommOffsets[j+start]+start;
            n1 = ((j == localCommNum-1) ? end : localCommOffsets[j+start+1]+start);
            GraphElem g = edges[n0];
            GraphElem2 g_id = search_ranges(vertex_per_device, g);
            GraphElem destCommId = commIdsPtr[g_id.x][g_id.y];
            if(destCommId == myCommId)
            {
                for(GraphElem k = n0+lane_id; k < n1; k+=WarpSize)
                    selfWeight += edgeWeights[k];
                for(unsigned int i = WarpSize/2; i > 0; i/=2)
                    selfWeight += warp.shfl_down(selfWeight, i);
                break;
            }
        }
        //warp.sync();
        if(lane_id == 0)
        {
            selfWeight /= (2.*mass);
            GraphElem2 v_id = search_ranges(vertex_per_device, v);
            GraphWeight ac = commWeightsPtr[v_id.x][v_id.y]; 
            selfWeight -= ac*ac/(4*mass*mass);
            mod[warp_id+BlockSize/WarpSize*blockIdx.x] += selfWeight;
        }
        //warp.sync();
        start = end;
    }
}
#if 0
template<const int BlockSize, const int WarpSize>
__global__
void compute_modularity_reduce_kernel
(
    GraphWeight*  __restrict__ mod,
    GraphElem*    __restrict__ edges,
    GraphWeight*  __restrict__ edgeWeights,
    GraphElem*    __restrict__ indices,
    GraphElem**    __restrict__ commIds,
    GraphWeight**  __restrict__ commWeights,
    GraphElem* localCommOffsets,
    GraphElem* localCommNums,
    const GraphWeight mass,
    const GraphElem v_base,
    const GraphElem e_base,
    const GraphElem nv
)
{
    //__shared__ GraphWeight self_shared[BlockSize/WarpSize];
    GraphWeight selfWeight;

    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    const unsigned warp_id      = threadIdx.x / WarpSize;
    const unsigned lane_id      = threadIdx.x & (WarpSize-1);

    GraphElem v1 = (nv + (BlockSize/WarpSize)*gridDim.x-1)/((BlockSize/WarpSize)*gridDim.x);
    GraphElem v0 = (warp_id + (BlockSize/WarpSize)*blockIdx.x)*v1;

    v1 += v0;
    if(v1 > nv) v1 = nv;
    v0 += v_base;
    v1 += v_base;
    GraphElem start, end;
    start = indices[v0]-e_base;
    
    for(GraphElem v = v0; v < v1; ++v)
    {
        selfWeight = 0;

        if(lane_id == 0x00)
            end = indices[v+1]-e_base;
        end = warp.shfl(end, 0);
        
        GraphElem localCommNum = localCommNums[v-v_base];
        GraphElem myCommId = commIds[0][v];
        //loop throught unique community ids
        for(GraphElem j = 0; j < localCommNum; ++j)
        {
            GraphElem n0, n1;
            n0 = localCommOffsets[j+start]+start;
            n1 = ((j == localCommNum-1) ? end : localCommOffsets[j+start+1]+start);
            GraphElem destCommId = commIds[0][edges[n0]];
            if(destCommId == myCommId)
            {
                for(GraphElem k = n0+lane_id; k < n1; k+=WarpSize)
                    selfWeight += edgeWeights[k];
                for(unsigned int i = WarpSize/2; i > 0; i/=2)
                    selfWeight += warp.shfl_down(selfWeight, i);
                break;
            }
        }
        //warp.sync();
        if(lane_id == 0)
        {
            //selfWeight = self_shared[warp_id];
            selfWeight /= (2.*mass);
            GraphWeight ac = commWeights[0][v]; 
            selfWeight -= ac*ac/(4*mass*mass);
            mod[warp_id+BlockSize/WarpSize*blockIdx.x] += selfWeight;
        }
        //warp.sync();
        start = end;
    }
}
#endif

template<const int BlockSize, const int WarpSize>
void compute_modularity_reduce_cuda
(
    GraphWeight*  mod,
    GraphElem*    edges,
    GraphWeight*  edgeWeights,
    GraphElem*    indices,
    GraphElem*    commIds,
    GraphElem**   commIdsPtr,
    GraphWeight** commWeightsPtr,
    GraphElem*    localCommOffsets,
    GraphElem*    localCommNums,
    const GraphWeight& mass,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    const GraphElem& V0,
    GraphElem* vertex_per_device,
    cudaStream_t stream = 0
)
{
    GraphElem nv = v1 - v0;
    long long nblocks = (nv+(BlockSize/WarpSize)-1)/(BlockSize/WarpSize);
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

    CudaLaunch((compute_modularity_reduce_kernel<BlockSize, WarpSize><<<nblocks, BlockSize, 0, stream>>>
    (mod, edges, edgeWeights, indices, commIds, commIdsPtr, commWeightsPtr, 
     localCommOffsets, localCommNums, mass, v0, e0, nv, V0, vertex_per_device)));

//     CudaLaunch((compute_modularity_reduce_kernel<BlockSize, WarpSize><<<nblocks, BlockSize, 0, stream>>>
//    (mod, edges, edgeWeights, indices, commIdsPtr, commWeightsPtr, localCommOffsets, localCommNums, mass, v0, e0, nv)));
}

//function instantialized
template void compute_modularity_reduce_cuda<BLOCKDIM02, WARPSIZE>
(
    GraphWeight*  mod,
    GraphElem*    edges,
    GraphWeight*  edgeWeights,
    GraphElem*    indices,
    GraphElem*    commIds,
    GraphElem**   commIdsPtr,
    GraphWeight** commWeightsPtr,
    GraphElem*    localCommOffsets,
    GraphElem*    localCommNums,
    const GraphWeight& mass,
    const GraphElem& v0,
    const GraphElem& v1,
    const GraphElem& e0,
    const GraphElem& e1,
    const GraphElem& V0,
    GraphElem* nv_per_device,
    cudaStream_t stream
);

GraphWeight compute_modularity_cuda
(
    GraphWeight* mod,
    const GraphElem& nv,
    cudaStream_t stream = 0
)
{
    CudaLaunch((reduce_vector_kernel<WARPSIZE><<<1, WARPSIZE*WARPSIZE, 0, stream>>>(mod, nv)));

    GraphWeight m;
    CudaMemcpyDtoH(&m, mod, sizeof(GraphWeight));
    return m;
}

#ifdef MULTIPHASE
template<const int BlockSize>
__global__
void fill_vertex_index_kernel
(
    GraphElem* __restrict__ vertex_index,
    const GraphElem nv,
    const GraphElem V0
)
{
    for(GraphElem i = threadIdx.x+BlockSize*blockIdx.x+V0; i < V0+nv; i += BlockSize*gridDim.x)
        vertex_index[i-V0] = i; 
}

void fill_vertex_index_cuda
(
    GraphElem* vertex_index,
    const GraphElem& nv,
    const GraphElem& V0,
    cudaStream_t stream = 0
)
{
    long long nblocks = (nv + BLOCKDIM03-1)/BLOCKDIM03;
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;
    CudaLaunch((fill_vertex_index_kernel<BLOCKDIM03><<<nblocks, BLOCKDIM03, 0, stream>>>
    (vertex_index, nv, V0)));    
}
/*
template<const int WarpSize>
__global__
void build_new_vertex_offset_kernel
( 
    GraphElem* __restrict__ vertexOffsets,
    GraphElem* __restrict__ newNv,
    GraphElem* __restrict__ commIds, 
    const GraphElem nv
)
{
    volatile GraphElem target = commIds[0];
    volatile GraphElem num = 0;
    volatile GraphElem count = 0;
    const unsigned warp_id = threadIdx.x & (WarpSize-1);
    while(count < nv)
    {
        volatile GraphElem change = 0;
        if(warp_id+count < nv)
        {
            if(commIds[warp_id+count]==target)
                change = 1;
        }
        for(unsigned i = WarpSize/2; i > 0; i/=2)
            change += __shfl_down_sync(0xffffffff, change, i, WarpSize);

        change = __shfl_sync(0xffffffff, change, 0);
        count += change;
        if(change < WarpSize)
        {
            if(warp_id == 0)
                vertexOffsets[num] = count;
            num++;
            if(count < nv)
                target = commIds[count];
        }
        //__syncthreads();
    } 
    if(warp_id == 0)
        *newNv = num;
}

template<const int BlockSize>
__global__
void build_new_commids_kernel
(
    GraphElem* __restrict__ commIds,
    GraphElem* __restrict__ vertexIds,
    GraphElem* __restrict__ vertexOffsets,
    GraphElem* __restrict__ nv_
)
{
    __shared__ GraphElem ranges[2];
    const GraphElem nv = *nv_;
    GraphElem start, end;    
    for(GraphElem i = blockIdx.x; i < nv; i += gridDim.x)
    {
        if(threadIdx.x == 0)
        {
            if(i == 0)
                ranges[0] = 0;
            else 
                ranges[0] = vertexOffsets[i-1];
            ranges[1] = vertexOffsets[i];
        } 
        __syncthreads();

        start = ranges[0];
        end = ranges[1];
        for(GraphElem j = start + threadIdx.x; j < end; j += BlockSize)
        {
            GraphElem v = vertexIds[j];
            commIds[v] = i;
        }
        __syncthreads();
    }
}

GraphElem build_new_vertex_id_cuda
( 
    GraphElem* commIds,
    GraphElem* vertexOffsets,
    GraphElem* newNv, 
    GraphElem* vertexIds, 
    const GraphElem& nv,
    cudaStream_t stream = 0
)
{
    CudaLaunch((build_new_vertex_offset_kernel<WARPSIZE><<<1, WARPSIZE, 0, stream>>>
    (vertexOffsets, newNv, commIds, nv))); 
    CudaLaunch((build_new_commids_kernel<BLOCKDIM01><<<dim3(MAX_GRIDDIM), BLOCKDIM01, 0, stream>>>
    (commIds, vertexIds, vertexOffsets, newNv)));
    GraphElem n;
    CudaMemcpyDtoH(&n, newNv, sizeof(GraphElem));
    return n;
}

template<const int BlockSize>
__global__
void compress_edges_kernel
(
    GraphElem*   __restrict__ edges, 
    GraphWeight* __restrict__ edgeWeights, 
    GraphElem*   __restrict__ numEdges, 
    GraphElem*   __restrict__ indices,
    GraphElem*   __restrict__ commIds,
    GraphElem*   localCommOffsets, 
    GraphElem*   localCommNums, 
    const GraphElem v_base, 
    const GraphElem e_base, 
    const GraphElem nv 
)
{
    for(GraphElem v = threadIdx.x+BlockSize*blockIdx.x; v < nv; v += BlockSize*gridDim.x)
    {
        GraphElem num = localCommNums[v];
        GraphElem start, end;
        start = indices[v+v_base+0]-e_base;
        end   = indices[v+v_base+1]-e_base;
        for(GraphElem i = 0; i < num; ++i)
        {
            GraphElem n0 = localCommOffsets[start+i]+start;
            GraphElem n1 = ((i == num-1) ? end : localCommOffsets[start+i+1]+start);
            GraphElem myId = commIds[edges[n0]];
            GraphWeight w = 0;
            for(GraphElem j = n0; j < n1; ++j)
                w += edgeWeights[j];
            edges[i+start] = myId;
            edgeWeights[i+start] = w;        
        }
        numEdges[v+v_base] = num;
    }
}

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
)
{
    GraphElem nv = v1-v0;
    long long nblocks = (nv+BLOCKDIM02-1)/BLOCKDIM02;
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;
    CudaLaunch((compress_edges_kernel<BLOCKDIM02><<<nblocks, BLOCKDIM02, 0, stream>>>
    (edges, edgeWeights, numEdges, indices, commIds, localCommOffsets, localCommNums, v0, e0, nv)));
}

template<typename T, const int BlockSize>
__global__
void sort_vector_kernel
(
    T* __restrict__ dest,
    T* __restrict__ src,
    GraphElem* __restrict__ orders,
    const GraphElem nv
)
{
    for(GraphElem i = threadIdx.x+BlockSize*blockIdx.x; i < nv; i += BlockSize*gridDim.x)
        dest[i] = src[orders[i]];
}


template<typename T>
void sort_vector_cuda
(
    T* dest, 
    T* src,
    GraphElem* orders,
    const GraphElem& nv,
    cudaStream_t stream = 0
)
{
    long long nblocks = (nv+BLOCKDIM03-1)/BLOCKDIM03;
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;
    CudaLaunch((sort_vector_kernel<T, BLOCKDIM03><<<nblocks, BLOCKDIM03, 0, stream>>>
    (dest, src, orders, nv)));
}

template
void sort_vector_cuda<GraphElem>
(
    GraphElem* src,
    GraphElem* dest,
    GraphElem* orders,
    const GraphElem& nv,
    cudaStream_t stream
);

template<const int BlockSize>
__global__
void compress_edge_ranges_kernel
(
    GraphElem* __restrict__ buffer,
    GraphElem* __restrict__ indices, 
    GraphElem* __restrict__ vertexOffsets, 
    const GraphElem nv
)
{
    for(GraphElem i = threadIdx.x+BlockSize*blockIdx.x; i < nv; i += BlockSize*gridDim.x)
    {
        //GraphElem offset = vertexOffsets[i];
        buffer[i] = indices[vertexOffsets[i]];
    }
}

void compress_edge_ranges_cuda
(
    GraphElem* indices,
    GraphElem*  buffer, 
    GraphElem* vertexOffsets, 
    const GraphElem& nv,
    cudaStream_t stream = 0
)
{
    long long nblocks = (nv+BLOCKDIM02-1)/BLOCKDIM02;
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

    CudaLaunch((compress_edge_ranges_kernel<BLOCKDIM02><<<nblocks, BLOCKDIM02, 0, stream>>>
    (buffer, indices, vertexOffsets, nv)));

    CudaLaunch((copy_vector_kernel<GraphElem, BLOCKDIM02><<<nblocks, BLOCKDIM02, 0, stream>>>(indices+1, buffer, nv)));
}*/
#endif

//kernels not used in the real Louvain implementations
template<const int WarpSize, const int BlockSize>
__global__
void max_vertex_weights_kernel 
(
    GraphWeight* __restrict__ maxVertexWeights,
    GraphWeight* __restrict__ edgeWeights,
    GraphElem*   __restrict__ edge_indices,
    const GraphElem v_base,   
    const GraphElem e_base,
    const GraphElem nv 
)
{
    __shared__ GraphElem ranges[(BlockSize/WarpSize)*2];

    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(cg::this_thread_block());

    const unsigned warp_tid = threadIdx.x & (WarpSize-1);

    GraphElem* t_ranges = &ranges[threadIdx.x/WarpSize*2];

    GraphElem step = gridDim.x*BlockSize/WarpSize;
    GraphElem u0 = (threadIdx.x/WarpSize)+(BlockSize/WarpSize)*blockIdx.x+v_base;

    for(GraphElem u = u0; u < v_base+nv; u += step)
    {   
        if(warp_tid < 2)
            t_ranges[warp_tid] = edge_indices[u+warp_tid];
        warp.sync();
       
        GraphElem start = t_ranges[0]-e_base;
        GraphElem   end = t_ranges[1]-e_base;
        volatile GraphWeight w = 0.; 
        for(GraphElem e = start+warp_tid; e < end; e += WarpSize)
        {
            GraphWeight tmp = edgeWeights[e];
            w = (tmp > w) ? tmp : w;
        }
        //warp.sync();

        for(int i = warp.size()/2; i > 0; i/=2)
        {
            GraphWeight tmp = warp.shfl_down(w, i);
            w = (tmp > w) ? tmp : w;
        }
        if(warp_tid == 0) 
            maxVertexWeights[u] = w;
        //warp.sync();
    }
}

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
)
{
    GraphElem nv = v1-v0;
    long long nblocks = (nv + BLOCKDIM01/WARPSIZE-1)/(BLOCKDIM01/WARPSIZE); 
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;
    CudaLaunch((max_vertex_weights_kernel<WARPSIZE, BLOCKDIM01><<<nblocks, BLOCKDIM01, 0, stream>>>
    (maxVertexWeights, edgeWeights, indices, v0, e0, nv)));
}

template<const int BlockSize>
__global__
void scan_edges_kernel
(
    GraphElem* __restrict__ edges,
    Edge*      __restrict__ edgeList,
    const GraphElem ne
)
{
    GraphElem u0 = threadIdx.x + BlockSize*blockIdx.x;
    for(GraphElem i = u0; i < ne; i += BlockSize*gridDim.x)
    {
        Edge e = edgeList[i];
        edges[i] = e.tail_;
    }
}

void scan_edges_cuda
(
    GraphElem* edges, 
    Edge* edgeList, 
    const GraphElem& e0, 
    const GraphElem& e1,
    cudaStream_t stream = 0
)
{
    GraphElem ne = e1-e0;
    long long nblocks = (ne+BLOCKDIM02-1)/BLOCKDIM02;
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;
    CudaLaunch((scan_edges_kernel<BLOCKDIM02><<<nblocks,BLOCKDIM02,0,stream>>>
    (edges, edgeList, ne)));
}

template<const int BlockSize>
__global__
void scan_edge_weights_kernel
(
    GraphWeight* __restrict__ edgeWeights,
    Edge*        __restrict__ edgeList,
    const GraphElem ne
)
{
    GraphElem u0 = threadIdx.x + BlockSize*blockIdx.x;
    for(GraphElem i = u0; i < ne; i += BlockSize*gridDim.x)
    {
        Edge e = edgeList[i];
        edgeWeights[i] = e.weight_;
    }
}

void scan_edge_weights_cuda
(
    GraphWeight* edgeWeights, 
    Edge* edgeList, 
    const GraphElem& e0, 
    const GraphElem& e1,
    cudaStream_t stream = 0
)
{
    long long ne = e1-e0;
    long long nblocks = (ne+BLOCKDIM02-1)/BLOCKDIM02;
    nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;
    CudaLaunch((scan_edge_weights_kernel<BLOCKDIM02><<<nblocks,BLOCKDIM02,0,stream>>>
    (edgeWeights, edgeList, ne)));
}


