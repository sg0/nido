#include <cstdlib>
#include <iostream>
#include <string>
#include <omp.h>
#include "graph.hpp"
#include "graph_gpu.hpp"
#include "louvain_gpu.hpp"
#include "types.hpp"
#include "cuda_wrapper.hpp"

#include <unistd.h>
#include <fstream>
#include <sstream>

int main(int argc, char** argv)
{
    using namespace std;
    
    Int nv;
    Graph* graph = nullptr;
    
    std::string inputFileName, outFileName, randGraphArgs, partArgs;
    int part_on_device = 1;
    int part_on_batch = 1;
    bool part_select = false;
    bool is_colored = false;
    bool rand_graph = false;
    bool bin_graph = false;
    bool output_communities = false;
    int nbatches = DEFAULT_BATCHES;
    Int maxLoops = DEFAULT_ITERATIONS;
    Float tau = DEFAULT_THRESHOLD;

    int ret;
    optind = 1;
    bool help_text = false;

    while ((ret = getopt(argc, argv, "f:p:r:cb:i:t:ho:")) != -1) 
    {
      switch (ret) 
      {
        case 'f':
          bin_graph = true;
          inputFileName.assign(optarg);
          break;
        case 'p':
          part_select = true;
          partArgs.assign(optarg);
          break;
        case 'r':
          rand_graph = true;
          randGraphArgs.assign(optarg);
          break;
        case 'o':
          output_communities = true;
          outFileName.assign(optarg);
          break;
        case 't':
          tau = (Float)atof(optarg);
          break;
        case 'i':
          maxLoops = (Int)atoll(optarg);
          break;
        case 'b':
          nbatches = (Int)atoi(optarg);
          break;
        case 'c':
          is_colored = true;
          break;
        case 'h':
          std::cout << "Sample usage [1] (use real-world file): ./run_<NGPU>_<GPU_ARCH> [-f /path/to/binary/file.bin] (see README)" << std::endl;
          std::cout << "Sample usage [2] (use synthetic graph): ./run_<NGPU>_<GPU_ARCH> [-r <#vertices> <edge-factor>" << std::endl;
          help_text = true;
          break;
        default:
          std::cout << "Reached default -- please check the passed options." << std::endl;
          break;
      }
    }
  
    if (help_text)
      std::exit(EXIT_SUCCESS);

    if(rand_graph)
    {
      std::stringstream ss(randGraphArgs);
      std::string s;
      std::vector<std::string> args;

      while (std::getline(ss, s, ' '))
        args.push_back(s);

      if (args.size() != 2) 
      {
        std::cerr << "For random graphs, expecting (in this order): <#vertices> <edge-factor>" 
          << std::endl;
        exit(EXIT_FAILURE);
      }

      nv = std::stoi(args[0]);
      Int m0 = std::stoi(args[1]);

      graph = new Graph(nv, m0);

      args.clear();
    }
    else if(bin_graph)
        graph = new Graph(inputFileName);
    else
    {
      std::cerr << "Input graph arguments missing. Use -f or -r options." << std::endl;
      std::exit(-1);
    }


    // this is for influencing the way we 
    // compute partitions - default is set,
    // do not change unless you know what you 
    // are doing...
    if(part_select)
    {
      std::stringstream ss(partArgs);
      std::string s;
      std::vector<std::string> args;

      while (std::getline(ss, s, ' '))
        args.push_back(s);

      if (args.size() != 2) 
      {
        std::cerr << "Expecting 0/1 (in this order): <part-on-device?> <part-on-batch?>" 
          << std::endl;
        exit(EXIT_FAILURE);
      }

      part_on_device = std::stoi(args[0]);
      part_on_batch = std::stoi(args[1]);

      args.clear();
    }
    
    std::cout << "#Batches: " << nbatches << std::endl;
    std::cout << "#Threshold: " << tau << std::endl;
    std::cout << "Max #iterations: " << maxLoops << std::endl;
    std::cout << "#NGPUs: " << NGPU << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    GraphElem* new_orders = nullptr;
    if(is_colored)
    {
        double color_start = omp_get_wtime();
        new_orders = graph->coloring(); 
        double color_end = omp_get_wtime();
        std::cout << "Coloring Time: " << color_end-color_start << " s\n";
    }
    
    GraphGPU* graph_gpu = new GraphGPU(graph, nbatches, part_on_device, part_on_batch);
    LouvainGPU* louvain = new LouvainGPU(maxLoops, tau, nbatches);

    louvain->run(graph_gpu);

    if (output_communities)
      graph_gpu->dump_partition(outFileName, new_orders);

    if(is_colored)
        delete [] new_orders;
    delete louvain;
    delete graph_gpu;
    delete graph;
    
    for(int i = 0; i < NGPU; ++i)
    {
        CudaSetDevice(i);
        CudaCall(cudaDeviceReset());
    }
    return 0;
}
