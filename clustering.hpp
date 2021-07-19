#ifndef CLUSTERING_HPP
#define CLUSTERING_HPP
#include "cuda_wrapper.hpp"
#include "types.hpp"

class Clustering
{
  private:

    GraphElem nv_;
    GraphElem* commIds_;
    GraphElem* commIdsHost_;
    void singleton_partition();

  public:
    Clustering(const GraphElem& nv) : nv_(nv)
    {
        commIds_ = new GraphElem [nv_];
        CudaMallocHost(commIdsHost_, sizeof(GraphElem)*nv_);
        singleton_partition();
    }
    ~Clustering()
    {
        delete [] commIds_;
        CudaFreeHost(commIdsHost_);
    }
    void move_community_to_host
    (
        GraphElem* commIds, 
        const GraphElem& nv, 
        cudaStream_t stream = 0
    )
    {
        CudaMemcpyAsyncDtoH(commIdsHost_, commIds, sizeof(GraphElem)*nv, stream);
    }
    void update_clustering();
};
#endif
