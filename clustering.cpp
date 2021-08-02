#include <omp.h>
#include <cstdio>
#include "clustering.hpp"
void Clustering::singleton_partition()
{
    #pragma omp parallel for
    for(GraphElem i = 0; i < nv_; ++i)
        commIds_[i] = i;
}

void Clustering::update_clustering(GraphElem* commIdsHost)
{
    #pragma omp parallel for
    for(GraphElem i = 0; i < nv_; ++i)
    {
        GraphElem myId = commIdsHost_[commIds[i]];
        commIds_[i] = myId;
    }
}

void Clustering::dump_partition(const std::string& filename)
{
    FILE* pFile;
    pFile = fopen(filename.c_str(), "wt");
    if(pFile != nullptr)
    {
        for(GraphElem i = 0; i < nv_; ++i)
            fprintf(pFile, "%lld\n", (long long)commIds_[i]);
    }
    fclose(pFile);
}
