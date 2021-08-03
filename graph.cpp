#include <cstdlib>
#include <omp.h>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <fstream>
#include <cstdint>
#include <climits>
#include "graph.hpp"
#include "types.hpp"
#include "heap.hpp"

Int Graph::get_num_vertices()
{
    return totalVertices_;
}

Int Graph::get_num_edges()
{
    return totalEdges_;
}

Int Graph::get_num_adjacent_vertices(const Int& i)
{
    return indices_[i+1]-indices_[i];
}

Int* Graph::get_adjacent_vertices(const Int& i)
{
    Int first = indices_[i];
    return edges_+first; 
}

Float* Graph::get_adjacent_weights(const Int& i)
{
    Int first = indices_[i];
    return weights_+first;
}

Int* Graph::get_orders()
{
    return orders_;
}
/* 
Float* Graph::get_weights()
{
    return weights_;
}
*/
Float* Graph::get_weighted_orders()
{
    return weighted_orders_;
}

Int Graph::get_max_order()
{
    return max_order_;
}

void Graph::sort_edges(EdgeTuple* edgeList, const Int& num)
{
    Heap<EdgeTuple, EdgeTupleMin> heap(edgeList, num);
    edges_   = new Int[num];
    weights_ = new Float[num];
    for(Int i = 0; i < num; ++i)
    {
        EdgeTuple e = heap.pop_back();
        edges_[i] = e.y;
        weights_[i] = e.w;
    }
}

void Graph::create_random_network_ba(const Int& m0)
{
    EdgeTuple* edgeList = new EdgeTuple[2*totalVertices_*m0];
    std::default_random_engine rand_gen1, rand_gen2;
    std::normal_distribution<Float> gaussian(1.0,1.0);

    indices_ = new Int[totalVertices_+1];
    indices_[0] = 0;
    for(Int i = 0; i < totalVertices_; ++i)
        indices_[i+1] = 0;

    totalEdges_ = 0;
    for(int i = 0; i < m0; ++i)
    {
        Float w = fabs(gaussian(rand_gen1));
        edgeList[totalEdges_+0] = {0,   i+1, w};
        edgeList[totalEdges_+1] = {i+1, 0,   w};
        indices_[1]   += 1;
        indices_[i+2] += 1;
        totalEdges_ += 2;
    }
    for(Int i = m0+1; i < totalVertices_; ++i)
    {
        std::uniform_int_distribution<Int> uniform(0,totalEdges_-1);
        for(Int j = 0; j < m0; ++j)
        {
            EdgeTuple e  = edgeList[uniform(rand_gen2)];
            Float w = fabs(gaussian(rand_gen1));
            edgeList[totalEdges_+2*j+0] = {i, e.x, w};
            edgeList[totalEdges_+2*j+1] = {e.x, i, w};
            indices_[i+1]   += 1;
            indices_[e.x+1] += 1;
        }
        totalEdges_ += 2*m0;
    }
    for(Int i = 1; i <= totalVertices_; ++i)
        indices_[i] += indices_[i-1];
    sort_edges(edgeList, totalEdges_);
    delete [] edgeList;
}

/*
void Graph::reset_orders_weights()
{
    for(Int i = 0; i < totalVertices_; ++i)
    {
        orders_[i] = 0;
        weighted_orders_[i] = 0.;
        max_weights_[i] = 0.;
    }
}
*/

void Graph::neigh_scan_max_order()
{
    Int o;
    for(Int i = 0; i < totalVertices_; ++i)
        if((o = orders_[i]) > max_order_)
            max_order_ = o;    
}

Graph::Graph(const Int& totalVertices, const Int& m0) :
totalVertices_(totalVertices), max_order_(0), 
weighted_orders_(nullptr), max_weights_(nullptr), orders_(nullptr),
indices_(nullptr), totalEdges_(0), edges_(nullptr), 
weights_(nullptr)
{
    create_random_network_ba(m0);
    
    weighted_orders_ = new Float [totalVertices_];
    max_weights_ = new Float[totalVertices_];
    orders_  = new Int[totalVertices_];
    for(Int i = 0; i < totalVertices_; ++i)
    {
        weighted_orders_[i] = 0.;
        max_weights_[i] = 0.;
        orders_[i] = 0;
    }
    neigh_scan();
    neigh_scan_weights();
    neigh_scan_max_weight();
    neigh_scan_max_order();
}
/*
Graph::Graph(Graph* g)
{
    totalVertices_ = g->totalVertices_;
    totalEdges_ = g->totalEdges_;
    max_order_ = g->max_order_;

    weighted_orders_ = new Float[totalVertices_];
    std::copy(g->weighted_orders_, g->weighted_orders_+totalVertices_, 
    weighted_orders_);  

    max_weights_ = new Float[totalVertices_];
    std::copy();

    orders_ = new Int[totalVertices_];      
    std::copy();

    indices_ = new Int[totalVertices_+1];
    std::copy();

    totalEdges_ = graph->totalEdges_;

    edges_ = new Int[];
    std::copy();

    weights_ = new Float[];
    std::copy();   
}
*/
void Graph::neigh_scan()
{
    for(Int i = 0; i < totalVertices_; ++i)
    {
        Int start = indices_[i];
        Int end = indices_[i+1];
        for(Int j = start; j < end; ++j)
            orders_[i] += 1;
    }
}

void Graph::neigh_scan_weights()
{
    for(Int i = 0; i < totalVertices_; ++i)
    {
        Int start = indices_[i];
        Int end = indices_[i+1];
        for(Int j = start; j < end; ++j)
        {
            Float w = weights_[j];
            weighted_orders_[i] += w;
        }
    }
}

void Graph::neigh_scan_max_weight()
{
    for(Int i = 0; i < totalVertices_; ++i)
    {
        Int start = indices_[i];
        Int end = indices_[i+1];
        Float max = 0.;
        for(Int j = start; j < end; ++j)
        {
            Float w = weights_[j];
            if(max < w)
                max = w;
        }
        max_weights_[i] = max;
    }
}

void Graph::neigh_scan(const int& num_threads)
{
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for(Int i = 0; i < totalVertices_; ++i)
    {
        Int start = indices_[i];
        Int end = indices_[i+1];
        for(Int j = start; j < end; ++j)
            orders_[i] += 1;
    }
}

void Graph::neigh_scan_weights(const int& num_threads)
{
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for(Int i = 0; i < totalVertices_; ++i)
    {
        Int start = indices_[i];
        Int end = indices_[i+1];
        for(Int j = start; j < end; ++j)
        {
            Float w = weights_[j];
            weighted_orders_[i] += w;
        }
    }
}

void Graph::neigh_scan_max_weight(const int& num_threads)
{
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for(Int i = 0; i < totalVertices_; ++i)
    {
        Int start = indices_[i];
        Int end = indices_[i+1];
        Float max = 0;
        for(Int j = start; j < end; ++j)
        {
            Float w = weights_[j];
            if(max < w)
                max = w;
        }
        max_weights_[i] = max;
    }
}

void Graph::print_stats()
{
    std::vector<Int> pdeg(totalVertices_, 0);
    for (Int v = 0; v < totalVertices_; v++)
    {
        Int num = get_num_adjacent_vertices(v);
        pdeg[v] += num;
    }
            
    std::sort(pdeg.begin(), pdeg.end());
    Float loc = (Float)(totalVertices_ + 1)/2.0;
    Int median;
    if (fmod(loc, 1) != 0)
        median = pdeg[(Int)loc]; 
    else 
        median = (pdeg[(Int)floor(loc)] + pdeg[((Int)floor(loc)+1)]) / 2;
    Int spdeg = std::accumulate(pdeg.begin(), pdeg.end(), 0);
    Int mpdeg = *(std::max_element(pdeg.begin(), pdeg.end()));
    std::transform(pdeg.cbegin(), pdeg.cend(), pdeg.cbegin(),
                   pdeg.begin(), std::multiplies<Int>{});

    Int psum_sq = std::accumulate(pdeg.begin(), pdeg.end(), 0);

    Float paverage = (Float) spdeg / totalVertices_;
    Float pavg_sq  = (Float) psum_sq / totalVertices_;
    Float pvar     = std::abs(pavg_sq - (paverage*paverage));
    Float pstddev  = sqrt(pvar);

    std::cout << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Graph characteristics" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Number of vertices: " << totalVertices_ << std::endl;
    std::cout << "Number of edges: " << totalEdges_ << std::endl;
    std::cout << "Maximum number of edges: " << mpdeg << std::endl;
    std::cout << "Median number of edges: " << median << std::endl;
    std::cout << "Expected value of X^2: " << pavg_sq << std::endl;
    std::cout << "Variance: " << pvar << std::endl;
    std::cout << "Standard deviation: " << pstddev << std::endl;
    std::cout << "--------------------------------------" << std::endl;
}

Graph::Graph(const std::string& binfile):
totalVertices_(0), max_order_(0),
weighted_orders_(nullptr), max_weights_(nullptr), orders_(nullptr),
indices_(nullptr), totalEdges_(0), edges_(nullptr),
weights_(nullptr)
{
    using GraphElem = Int;
    using GraphWeight = Float;

    std::ifstream file;
    file.open(binfile.c_str(), std::ios::in | std::ios::binary); 

    if (!file.is_open()) 
    {
        std::cout << " Error opening file! " << std::endl;
        std::abort();
    }

    // read the dimensions 
    file.read(reinterpret_cast<char*>(&totalVertices_), sizeof(GraphElem));
    file.read(reinterpret_cast<char*>(&totalEdges_), sizeof(GraphElem));

    //weighted_orders_ = new GraphWeight [totalVertices_];
    //max_weights_ = new GraphWeight [totalVertices_];
    //orders_ = new GraphElem [totalVertices_];
     
    indices_ = new GraphElem [totalVertices_+1];
    //edges_ = new GraphElem [totalEdges_];
    //weights_ = new GraphWeight [totalEdges_];
    Edge* edges = new Edge [totalEdges_];

    uint64_t tot_bytes=(totalVertices_+1)*sizeof(GraphElem);
    ptrdiff_t offset = 2*sizeof(GraphElem);

    if (tot_bytes < INT_MAX)
        file.read(reinterpret_cast<char*>(&indices_[0]), tot_bytes);
    else 
    {
        int chunk_bytes = INT_MAX;
        uint8_t *curr_pointer = (uint8_t*) &indices_[0];
        uint64_t transf_bytes = 0;

        while (transf_bytes < tot_bytes)
        {
            file.read(reinterpret_cast<char*>(&curr_pointer[offset]), chunk_bytes);
            transf_bytes += chunk_bytes;
            offset += chunk_bytes;
            curr_pointer += chunk_bytes;

            if ((tot_bytes - transf_bytes) < INT_MAX)
                chunk_bytes = tot_bytes - transf_bytes;
        } 
    }    

    if(indices_[totalVertices_] - indices_[0] != totalEdges_)
    {
        std::cerr << "Error format in the file\n";
        std::abort();
    }
 
    tot_bytes = totalEdges_*(sizeof(Edge));
    offset = 2*sizeof(GraphElem) + (totalVertices_+1)*sizeof(GraphElem) 
           + indices_[0]*(sizeof(Edge));

#if defined(GRAPH_FT_LOAD)
    ptrdiff_t currpos = file.tellg();
    ptrdiff_t idx = 0;
    GraphElem* vidx = new GraphElem [totalVertices_];

    const int num_sockets = (GRAPH_FT_LOAD == 0) ? 1 : GRAPH_FT_LOAD;
    printf("Read file from %d sockets\n", num_sockets);
    int n_blocks = num_sockets;

    //GraphElem NV_blk_sz = totalVertices_ / n_blocks;
    //GraphElem tid_blk_sz = omp_get_num_threads() / n_blocks;
    GraphElem NV_blk_sz = (totalVertices_+n_blocks-1) / n_blocks;
    GraphElem tid_blk_sz = omp_get_num_threads() / n_blocks;
    #pragma omp parallel
    {
        for (int b=0; b<n_blocks; b++) 
        {
            long NV_beg = b * NV_blk_sz;
            long NV_end = std::min(totalVertices_, ((b+1) * NV_blk_sz) );
            int tid_doit = b * tid_blk_sz;

            if (omp_get_thread_num() == tid_doit) 
            {
                // for each vertex within block
                for (GraphElem i = NV_beg; i < NV_end ; i++) 
                {
                    // ensure first-touch allocation
                    // read and initialize using your code
                    vidx[i] = idx;
                    const GraphElem vcount = indices_[i+1] - indices_[i];
                    idx += vcount;
                    file.seekg(currpos + vidx[i] * sizeof(Edge), std::ios::beg);
                    //file.read(reinterpret_cast<char*>(&edges_[vidx[i]]), sizeof(Edge) * (vcount));
                    file.read(reinterpret_cast<char*>(&edges[vidx[i]]), sizeof(Edge) * (vcount));
                }
            }
        }
    }
    delete [] vidx;
#else
    if (tot_bytes < INT_MAX)
        file.read(reinterpret_cast<char*>(&edges[0]), tot_bytes);
    else 
    {
        int chunk_bytes=INT_MAX;
        uint8_t *curr_pointer = (uint8_t*)&edges[0];
        uint64_t transf_bytes = 0;

        while (transf_bytes < tot_bytes)
        {
            file.read(reinterpret_cast<char*>(&curr_pointer[offset]), tot_bytes);
            transf_bytes += chunk_bytes;
            offset += chunk_bytes;
            curr_pointer += chunk_bytes;

            if ((tot_bytes - transf_bytes) < INT_MAX)
                chunk_bytes = (tot_bytes - transf_bytes);
        } 
    } 
    file.close();
#endif

    edges_ = new GraphElem [totalEdges_];
    weights_ = new GraphWeight [totalEdges_];
    //std::cout << totalEdges_ << " " << indices_[0] << " " << indices_[totalVertices_] << std::endl;
    for(GraphElem i = 0; i < totalEdges_; ++i)
    {
        Edge e = edges[i];
        edges_[i] = e.tail_;
        weights_[i] = e.weight_;
    }
    delete [] edges;

    for(GraphElem i=1;  i < totalVertices_+1; i++)
        indices_[i] -= indices_[0];
    indices_[0] = 0;

    weighted_orders_ = new GraphWeight [totalVertices_];
    max_weights_ = new GraphWeight [totalVertices_];
    orders_ = new GraphElem [totalVertices_];

    for(Int i = 0; i < totalVertices_; ++i)
    {
        weighted_orders_[i] = 0.;
        max_weights_[i] = 0.;
        orders_[i] = 0;
    }

    neigh_scan();
    neigh_scan_weights();
    neigh_scan_max_weight();
    neigh_scan_max_order();
    print_stats();
}

Int* Graph::get_index_ranges()
{
    return indices_;
}

Float* Graph::get_edge_weights()
{
    return weights_;
}

Int* Graph::get_edges()
{
    return edges_;
}

#if 0
void Partition::destroy_partition()
{
    delete [];
    delete [];
}

void Partition::singleton_partition()
{
    destroy_partition();
        
}

Partition::Partition(const Graph& g): graph(&g), commMap(NULL), 
community(NULL)
{
    singleton_partition();
}

void Partition::set_graph(const Graph& g)
{ 
    graph = &g;
    singleton_partition();
}

long Partition::get_comm_id(const long& i)
{
    return commMap[i];
}

Community* Partition::get_community(const long& i)
{
    return community[i]; 
}
#endif
