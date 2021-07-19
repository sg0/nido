#ifndef CLUSTERING_HPP
#define CLUSTERING_HPP

#include "types.hpp"

struct Vertex
{
    GraphElem v;
    struct Vertex* next;
};

typedef struct Vertex Vertex;

class Clustering
{
  private:

    GraphElem nv_;
    Vertex** vertex_list_;
    Vertex** community_list_;

    GraphElem* vertexIds_, *vertexOffsets_;

  public:
    Clustering(const GraphElem& nv) : nv_(nv)
    {
        vertex_list_    = new Vertex* [nv_];
        community_list_ = new Vertex* [nv_];
        vertexIds_      = new GraphElem [nv_];
        vertexOffsets_  = new GraphElem [nv_];

        for(GraphElem i = 0; i < nv_; ++i)
        {
            Vertex* vertex = new Vertex;
            vertex->v = i; vertex->next = nullptr;
            vertex_list_[i] = vertex;
        }
        for(GraphElem i = 0; i < nv_; ++i)
            community_list_[i] = nullptr;
    }
    ~Clustering();
    void singleton_partition()
    {
        for(GraphElem i = 0; i < nv_; ++i)
            community_list_[i] = vertex_list_[i];
    }
    void aggregate_vertex
    (
        GraphElem* vertexIds_dev, 
        GraphElem* vertexOffsets_dev,
        const GraphElem& newNv
    );
};
#endif
