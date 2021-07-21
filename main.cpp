#include <cstdlib>
#include <iostream>
#include <string>
#include "graph.hpp"
#include "graph_gpu.hpp"
#include "louvain_gpu.hpp"
#include "types.hpp"
#include "cuda_wrapper.hpp"

int main(int argc, char** argv)
{
    using namespace std;

    Int nv;
    Graph* graph = nullptr;
    int pos = 0;
    if(std::string(argv[1]) == "-r")
    {
        nv = atoi(argv[2]);
        Int m0 = atoi(argv[3]);
        graph = new Graph(nv, m0);
        pos = 4;
    }
    else if(std::string(argv[1]) == "-f")
    {
        graph = new Graph(std::string(argv[2]));
        pos = 3;
    }

    int dev_id = atoi(argv[pos]);
    CudaCall(cudaSetDevice(dev_id));

    Int maxLoops = (Int)atoll(argv[pos+1]);
    Float tau = (Float)atof(argv[pos+2]);
    Int nbatches = (Int)atoll(argv[pos+3]);
    GraphGPU* graph_gpu = new GraphGPU(graph);
    
    //Partition* partition = new Partition(graph);
    LouvainGPU* louvain = new LouvainGPU(maxLoops, tau, nbatches);

    louvain->run(graph_gpu);
    //louvain->dump_partition(std::string(argv[pos+4]));

    delete louvain;
    delete graph_gpu;
    delete graph;
    CudaCall(cudaDeviceReset());
    return 0;
}
