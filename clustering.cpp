#include <omp.h>
#include "clustering.hpp"
#include "cuda_wrapper.hpp"
Clustering::~Clustering()
{
    #pragma omp parallel
    for(GraphElem i = 0; i < nv_; ++i)
    {
        Vertex* ptr = vertex_list_[i];
        while(ptr != nullptr)
        {
            Vertex* tmp = ptr->next;
            delete ptr;
            ptr = tmp;
        } 
    }
    delete [] vertex_list_;
    delete [] community_list_;
    delete [] vertexIds_;
    delete [] vertexOffsets_;
}

void Clustering::aggregate_vertex
(
    GraphElem* vertexIds_dev, 
    GraphElem* vertexOffsets_dev, 
    const GraphElem& newNv
)
{
     cudaStream_t cuStreams[2];

     cudaStreamCreate(&cuStreams[0]);
     cudaStreamCreate(&cuStreams[1]);

     CudaMemcpyAsyncDtoH(vertexIds_, vertexIds_dev, sizeof(GraphElem)*newNv, cuStreams[0]);
     CudaMemcpyAsyncDtoH(vertexOffsets_, vertexOffsets_dev, sizeof(GraphElem)*newNv, cuStreams[1]);

     CudaDeviceSynchronize();

     #pragma omp parallel
     for(GraphElem i = 0; i < newNv; ++i)
     {
         GraphElem n0, n1;
         if(i == 0) 
             n0 = 0;
         else 
             n0 = vertexOffsets_[i-1];
         n1 = vertexOffsets_[i];

         Vertex* ptr = community_list_[i];
         for(GraphElem j = n0; j < n1; ++j)
         {
             GraphElem id = vertexIds_[j];
             if(ptr == nullptr)
                 community_list_[i] = vertex_list_[id];
             else
             {    
                 while(ptr->next != nullptr)
                     ptr = ptr->next;
             
                 ptr->next = vertex_list_[id];
             }
         }
     }
     nv_ = newNv;
     #pragma omp parallel
     for(GraphElem i = 0; i < nv_; ++i)
     {
         vertex_list_[i] = community_list_[i]; 
         community_list_[i] = nullptr;
     }

    cudaStreamDestroy(cuStreams[0]);
    cudaStreamDestroy(cuStreams[1]);
}

