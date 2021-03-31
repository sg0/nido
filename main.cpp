#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#include <cassert>
#include <cstdlib>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "clustering.hpp"

#if defined(SCOREP_USER_ENABLE)
#include <scorep/SCOREP_User.h>
#endif

static std::string inputFileName;
static int me, nprocs;
static int ranksPerNode = 1;
static GraphElem nvRGG = 0;
static int generateGraph = 0;

static GraphWeight randomEdgePercent = 0.0;
static bool readBalanced = false;
static bool randomNumberLCG = false;

// parse command line parameters
static void parseCommandLine(int argc, char** argv);

int main(int argc, char **argv)
{
    double t0, t1, td, td0, td1;

#ifdef DISABLE_THREAD_MULTIPLE_CHECK
    MPI_Init(&argc, &argv);
#else  
    int max_threads;

    max_threads = omp_get_max_threads();

    if (max_threads > 1) 
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        if (provided < MPI_THREAD_MULTIPLE) 
        {
            std::cerr << "MPI library does not support MPI_THREAD_MULTIPLE." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -99);
        }
    } 
    else 
    {
        MPI_Init(&argc, &argv);
    }
#endif

#if defined(SCOREP_USER_ENABLE)
    SCOREP_RECORDING_OFF();
#endif
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    // command line options
    MPI_Barrier(MPI_COMM_WORLD);
    parseCommandLine(argc, argv);
 
    Graph* g = nullptr;
    
    td0 = MPI_Wtime();

    // generate graph only supports RGG as of now
    if (generateGraph) 
    { 
        GenerateRGG gr(nvRGG);
        g = gr.generate(randomNumberLCG, true /*isUnitEdgeWeight*/, randomEdgePercent);
    }
    else 
    {   // read input graph
#ifndef SSTMAC
        BinaryEdgeList rm;
        if (readBalanced == true)
        {
            if (me == 0)
            {
                std::cout << std::endl;
                std::cout << "Trying to balance the edge distribution while reading: " << std::endl;
                std::cout << inputFileName << std::endl;
            }
            g = rm.read_balanced(me, nprocs, ranksPerNode, inputFileName);
        }
        else
            g = rm.read(me, nprocs, ranksPerNode, inputFileName);
#else
#warning "SSTMAC is defined: Trying to load external graph binaries will FAIL."
#endif
    }

#if defined(PRINT_GRAPH_EDGES)        
    g->print();
#endif
    g->print_dist_stats();
    assert(g != nullptr);

    MPI_Barrier(MPI_COMM_WORLD);

    td1 = MPI_Wtime();
    td = td1 - td0;

    double tdt = 0.0;
    MPI_Reduce(&td, &tdt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (me == 0)  
    {
        if (!generateGraph)
            std::cout << "Time to read input file and create distributed graph (in s): " 
                << tdt << std::endl;
        else
            std::cout << "Time to generate distributed graph of " 
                << nvRGG << " vertices (in s): " << tdt << std::endl;
    }
   
    GraphWeight mod;

    // Clustering instantiation
    Clustering cl(g);

    MPI_Barrier(MPI_COMM_WORLD);
#if defined(SCOREP_USER_ENABLE)
    SCOREP_RECORDING_ON();
    SCOREP_USER_REGION_BY_NAME_BEGIN("Clustering-First-Phase", SCOREP_USER_REGION_TYPE_COMMON);
    if (me == 0)
        SCOREP_USER_REGION_BY_NAME_BEGIN("TRACER_WallTime_Clustering", SCOREP_USER_REGION_TYPE_COMMON);
#endif
    t0 = MPI_Wtime();

    int iters = cl.run_louvain(mod);

#if defined(SCOREP_USER_ENABLE)
    if (me == 0)
        SCOREP_USER_REGION_BY_NAME_END("TRACER_WallTime_Clustering");
    SCOREP_USER_REGION_BY_NAME_END("Clustering-First-Phase");
    SCOREP_RECORDING_OFF();
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    
    t1 = MPI_Wtime();
    double p_tot = t1 - t0, t_tot = 0.0;
    
    MPI_Reduce(&p_tot, &t_tot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (me == 0) 
    {
        double avgt = (t_tot / nprocs);
        if (!generateGraph) 
        {
            std::cout << "-------------------------------------------------------" << std::endl;
            std::cout << "File: " << inputFileName << std::endl;
            std::cout << "-------------------------------------------------------" << std::endl;
        }
        std::cout << "-------------------------------------------------------" << std::endl;
#ifdef USE_32_BIT_GRAPH
        std::cout << "32-bit datatype" << std::endl;
#else
        std::cout << "64-bit datatype" << std::endl;
#endif
        std::cout << "-------------------------------------------------------" << std::endl;
        std::cout << "Average total time (in s), #Processes: " << avgt << ", " << nprocs << std::endl;
        std::cout << "Modularity, #Iterations: " << iter_mod << ", " << iters << std::endl;
        std::cout << "MODS (final modularity * average time): " << (iter_mod * avgt) << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;
#ifndef SSTMAC
        std::cout << "Resolution of MPI_Wtime: " << MPI_Wtick() << std::endl;
#endif
    }

    cl.clear(); 

    MPI_Barrier(MPI_COMM_WORLD);
   
    MPI_Finalize();

    return 0;
}

void parseCommandLine(int argc, char** const argv)
{
  int ret;
  optind = 1;

  while ((ret = getopt(argc, argv, "f:r:n:lp:b")) != -1) 
  {
      switch (ret) 
      {
          case 'f':
              inputFileName.assign(optarg);
              break;
          case 'b':
              readBalanced = true;
              break;
          case 'r':
              ranksPerNode = atoi(optarg);
              break;
          case 'n':
              nvRGG = atol(optarg);
              if (nvRGG > 0)
                  generateGraph = true; 
              break;
          case 'l':
              randomNumberLCG = true;
              break;
          case 'p':
              randomEdgePercent = atof(optarg);
              break;
          default:
              assert(0 && "Should not reach here!!");
              break;
      }
  }

  // warnings/info

  if (me == 0 && generateGraph && readBalanced) 
  {
      std::cout << "Balanced read (option -b) is only applicable for real-world graphs. "
          << "This option does nothing for generated (synthetic) graphs." << std::endl;
  } 
   
  // errors
  if (me == 0 && (argc == 1)) 
  {
      std::cerr << "Must specify some options." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
  
  if (me == 0 && !generateGraph && inputFileName.empty()) 
  {
      std::cerr << "Must specify a binary file name with -f or provide parameters for generating a graph." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
   
  if (me == 0 && !generateGraph && randomNumberLCG) 
  {
      std::cerr << "Must specify -n for graph generation using LCG." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  } 
   
  if (me == 0 && !generateGraph && (randomEdgePercent > 0.0)) 
  {
      std::cerr << "Must specify -n for graph generation first to add random edges to it." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  } 
  
  if (me == 0 && generateGraph && ((randomEdgePercent < 0.0) || (randomEdgePercent >= 100.0))) 
  {
      std::cerr << "Invalid random edge percentage for generated graph!" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
} // parseCommandLine
