#include "types.hpp"
#include "cuda_wrapper.hpp"
#include <algorithm>
#include <iostream>
#include <vector>
typedef struct Edge2
{
    GraphElem v, commIds;
} Edge2;
struct 
{
    bool operator()(Edge2 a, Edge2 b) const 
    { 
        return ((a.commIds != b.commIds) ? a.commIds < b.commIds : a.v < b.v); 
    }
} less;

void build_new_vertex_id_cpu
(
    GraphElem* commIdsHost, 
    GraphElem* vertexOffsets, 
    GraphElem* newNv, 
    GraphElem* vertexIds, 
    const GraphElem& nv
)
{
    Edge2* vec = new Edge2[nv];
    GraphElem* offsetsHost = new GraphElem[nv];
    CudaMemcpyDtoH(offsetsHost, vertexOffsets, sizeof(GraphElem)*nv);

    GraphElem* vertexIdsHost = new GraphElem[nv];
    CudaMemcpyDtoH(vertexIdsHost, vertexIds, sizeof(GraphElem)*nv);

    #pragma omp parallel
    for(GraphElem i = 0; i < nv; ++i)
        vec[i] = {i, commIdsHost[i]};
    std::sort(vec, vec+nv, less);

    GraphElem target = vec[0].commIds;
    //GraphElem offset = 0;
    std::vector<GraphElem> offsets;
    GraphElem offset = 0;
    for(offset = 0; offset < nv; ++offset)
    {
        if(vec[offset].commIds != target)
        {
            offsets.push_back(offset);
            target = vec[offset].commIds;
        }
        //offset++;
    }
    offsets.push_back(offset);
    GraphElem n = 0;
    CudaMemcpyDtoH(&n, newNv, sizeof(GraphElem));
    std::cout << n << " " << offsets.size() << std::endl;

    GraphElem err = 0;
    for(GraphElem i = 0; i < n; ++i)
        err += abs(offsets[i]-offsetsHost[i]);
        //std::cout << offsets[i] << " " << offsetsHost[i] << std::endl;
    std::cout << "error " << err << std::endl;

    //for(GraphElem i = 0; i < nv; ++i)
    //    std::cout << commIdsHost[vertexIdsHost[i]] << " " << vec[i].commIds << std::endl;
    delete [] vec;
    delete [] offsetsHost;
    delete [] vertexIdsHost;
    //exit(-1);
}
/*
void compress_edges_cpu(edgesHost_, edgeWeightsHost_, indicesHost_, numEdgesHost, edges, weights, v0, e0)
{*/

//}
