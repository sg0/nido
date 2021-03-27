#pragma once
#ifndef LOUVAIN_HPP
#define LOUVAIN_HPP

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>

#include <mpi.h>
#include <omp.h>

#include "graph.hpp"
#include "utils.hpp"

int run_louvain(Graph* g, GraphWeight lower = -1.0, GraphWeight thresh = 1.0E-06)
{
  const GraphElem nv = g->get_lnv();
  MPI_Comm gcomm = g->get_comm();

  GraphWeight constant_term = 1.0 / (GraphWeight)2.0 * g->get_sum_weights();
  GraphWeight prev_mod = lower;
  GraphWeight mod = -1.0;
  int iters = 0;
  
  while(true) 
  {
    iters++;

    MPI_Put();

    mod = louvain_iteration<<< >>>(g);

    if (mod - prev_mod < thresh)
        break;

    prev_mod = mod;
    if (prev_mod < lower)
        prev_mod = lower;
  } 

  MPI_Win_unlock_all(commwin);
  MPI_Win_free(&commwin);

  return iters;
}

__device__ inline
GraphWeight warpAllReduceSum(GraphWeight val) 
{
  for (int mask = warpSize/2; mask > 0; mask /= 2)
      val += __shfl_xor(val, mask);
  return val;
}

__inline__ __device__
GraphWeight blockReduceSum(GraphWeight val) 
{

  static __shared__ int shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__global__ inline
void deviceReduceBlockAtomicKernel(GraphWeight* in, GraphWeight* out, int N) 
{
  GraphWeight sum = GraphWeight(0);
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    sum += in[i];

  sum = blockReduceSum(sum);
  
  if (threadIdx.x == 0)
    atomicAdd(out, sum);
}

__global__ inline
void louvain_iteration(GraphElem lnv, GraphElem lne, 
        GraphElem* cluster, GraphElem* cluster_degree, 
        GraphWeight* cluster_weight)
{


}

GraphWeight compute_modularity(Graph* g)
{
  const GraphElem nv = g->get_lnv();
  MPI_Comm gcomm = g->get_comm();

  GraphWeight le_la_xx[2];
  GraphWeight e_a_xx[2] = {0.0, 0.0};
  GraphWeight le_xx = 0.0, la2_x = 0.0;

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(shared), shared(clusterWeight, localCinfo), \
  reduction(+: le_xx), reduction(+: la2_x) schedule(runtime)
#else
#pragma omp parallel for default(shared), shared(clusterWeight, localCinfo), \
  reduction(+: le_xx), reduction(+: la2_x) schedule(static)
#endif
  for (GraphElem i = 0; i < nv; i++) 
  {
    le_xx += g->cluster_weight_[i];
    la2_x += static_cast<GraphWeight>(g->cluster_degree_[i]) * static_cast<GraphWeight>(g->cluster_degree_[i]); 
  } 
  le_la_xx[0] = le_xx;
  le_la_xx[1] = la2_x;

#ifdef DEBUG_PRINTF  
  const double t0 = MPI_Wtime();
#endif

  MPI_Allreduce(le_la_xx, e_a_xx, 2, MPI_WEIGHT_TYPE, MPI_SUM, gcomm);

#ifdef DEBUG_PRINTF  
  const double t1 = MPI_Wtime();
#endif

  GraphWeight currMod = std::fabs((e_a_xx[0] * constantForSecondTerm) - 
      (e_a_xx[1] * constantForSecondTerm * constantForSecondTerm));

  return currMod;
}

#endif // LOUVAIN_HPP
