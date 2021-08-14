#ifndef LOUVAIN_GPU_HPP_
#define LOUVAIN_GPU_HPP_
#include "types.hpp"
#include "graph_gpu.hpp"

class LouvainGPU
{
  private:
    Int maxLoops_;//, nbatches_; 
    Float tol_;
    int nbatches_;

  public:
    LouvainGPU(const Int& maxLoops, const Float& tol, const int& nbatches) :
    maxLoops_(maxLoops), tol_(tol), nbatches_(nbatches) {};
 
    ~LouvainGPU() {};

    void run(GraphGPU*);
};
#endif
