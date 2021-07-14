#ifndef GRAPH_HPP_
#define GRAPH_HPP_
#include <list>
#include <string>
#include "types.hpp"

class Graph
{
  private:
    GraphElem   totalVertices_;
    GraphElem   max_order_;
    GraphWeight *weighted_orders_;
    GraphWeight *max_weights_;
    GraphElem   *orders_;

    GraphElem   *indices_;
    GraphElem   totalEdges_;
    GraphElem   *edges_;
    GraphWeight *weights_;

    //some helper function here
    void sort_edges(EdgeTuple*, const GraphElem&);
    void create_random_network_ba(const GraphElem& m0);
    void neigh_scan();
    void neigh_scan_weights();
    void neigh_scan_max_weight();
    void neigh_scan_max_order();

  public:      
    Graph(const GraphElem&, const GraphElem&);
    Graph(const std::string&);
    //Graph(Graph* g);
 
    ~Graph()
    {
        delete [] weighted_orders_;
        weighted_orders_ = nullptr;  
      
        delete [] max_weights_;
        max_weights_ = nullptr;

        delete [] orders_;
        orders_ = nullptr;

        delete [] indices_;
        indices_ = nullptr;

        delete [] edges_;
        edges_ = nullptr;

        delete [] weights_;
        weights_ = nullptr;
        
    }
    GraphElem*    get_adjacent_vertices(const GraphElem&);
    GraphWeight*  get_adjacent_weights(const GraphElem&);
    GraphElem*    get_orders();
    GraphWeight*  get_weighted_orders();
    GraphElem     get_num_adjacent_vertices(const GraphElem&);
    GraphElem     get_num_vertices();
    GraphElem     get_num_edges();
    GraphElem     get_max_order();

    GraphElem*    get_index_ranges();
    GraphWeight*  get_edge_weights();
    GraphElem*    get_edges();
    //print statistics
    void print_stats();

    //void reset_orders_weights();
    void neigh_scan(const int& num_threads);
    void neigh_scan_weights(const int& num_threads);
    void neigh_scan_max_weight(const int& num_threads);

    //void sort_edges_by_community_ids();    
};
#endif
