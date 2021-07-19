#include <omp.h>
#include "clustering.hpp"
void Clustering::singleton_partition()
{
    #pragma omp parallel for
    for(GraphElem i = 0; i < nv_; ++i)
        commIds_[i] = i;
}

void Clustering::update_clustering()
{
    #pragma omp parallel for
    for(GraphElem i = 0; i < nv_; ++i)
    {
        GraphElem myId = commIdsHost_[commIds_[i]];
        commIds_[i] = myId;
    }
}

