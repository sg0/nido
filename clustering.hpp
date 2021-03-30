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


class Clustering
{
    public:
        void nbcomm_create()
        {
            const GraphElem lnv = g_->get_lnv();
            for (GraphElem v = 0; v < lnv; v++)
            {
                GraphElem e0, e1;
                g_->edge_range(v, e0, e1);

                for (GraphElem e = e0; e < e1; e++)
                {
                    Edge const& edge = g_->get_edge(e);
                    const int owner = g_->get_owner(edge.tail_); 
                    if (owner != rank_)
                    {
                        if (std::find(targets_.begin(), targets_.end(), owner) 
                                == targets_.end()
                                && std::find(sources_.begin(), sources_.end(), owner) 
                                == sources_.end())
                        {
                            targets_.push_back(owner);
                            sources_.push_back(owner);
                        }
                    }
                }
            }

            MPI_Dist_graph_create_adjacent(comm_, targets_.size(), targets_.data(), 
                    MPI_UNWEIGHTED, targets_.size(), targets_.data(), MPI_UNWEIGHTED, 
                    MPI_INFO_NULL, 0 /*reorder ranks*/, &nbcomm_);

            // checking...
            int weighted;
            MPI_Dist_graph_neighbors_count(nbcomm_, &indegree_, &outdegree_, &weighted);
            
            assert(indegree_ == targets_.size());
            assert(outdegree_ == targets_.size());
        }
        
        void nbcomm_destroy()
        { MPI_Comm_free(&nbcomm_); }


int run_louvain(Graph* g, GraphWeight lower = -1.0, GraphWeight thresh = 1.0E-06)
{
  const GraphElem nv = g->get_lnv();
  MPI_Comm gcomm = g->get_comm();

  GraphWeight prev_mod = lower;
  GraphWeight mod = -1.0;
  int iters = 0;
  
  while(true) 
  {
    iters++;

    MPI_Put();

    if (iters == 1)
        mod = louvain_iteration_first(g);
    else
        mod = louvain_iteration_next(g);

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

GraphWeight louvain_iteration(Graph *g, int me)
{
  GraphWeight le_xx = 0.0, la2_x = 0.0, local_mod, mod;
  const GraphElem lnv = g->get_lnv();
  GraphWeight tot_weights = g->get_sum_weights();
  GraphWeight constant_term = 1.0 / (GraphWeight)2.0 * tot_weights;
  
  std::vector<GraphElem> cluster_degree(lnv);
  std::vector<GraphWeight> cluster_weights(lnv);
  std::copy(g->cluster_degree_.begin(), g->cluster_degree_.end(), cluster_degree.begin());
  std::copy(g->cluster_weight_.begin(), g->cluster_weight_.end(), cluster_weight.begin());

  // local modularity
  for (GraphElem i = 0; i < lnv; i++) 
  {
      le_xx += cluster_weight[i];
      la2_x += static_cast<GraphWeight>(cluster_degree_[i]) * static_cast<GraphWeight>(cluster_degree_[i]); 
  } 
  
  mod = std::fabs((le_xx * constant_term) - (la2_x * constant_term * constant_term));

  GraphElem cluster_id = 0;

  // calculate delta-q and determine target community
  for (GraphElem i = 0; i < lnv; i++)
  {
      g->edge_range(i, e0, e1);
      GraphElem target_cluster = -1;
      GraphWeight upd_mod = 0.0;
      
      for (GraphElem e = e0; e < e1; e++)
      {
          GraphWeight la2_xup = la2_x;
          Edge const& edge = g->get_edge(e);
          
          if (g->owner(edge.tail_) == me)
          {
              GraphElem curr_degree = cluster_degree[i];
              curr_degree += cluster_degree[g->global_to_local(edge.tail_)];
              la2_xup += (curr_degree * curr_degree) - (cluster_degree[i] * cluster_degree[i]);
          }
          else
              la2_xup -= cluster_degree[i] * cluster_degree[i];

          upd_mod = std::fabs((le_xx * constant_term) - (la2_xup * constant_term * constant_term));

          if (upd_mod - mod > 0.0)
              target_cluster = edge.tail_;
      }

      // adjust cluster
      if (target_cluster != -1)
      {
          cluster_id++;

          g->cluster_[i] = cluster_id;
          g->cluster_degree_[i] = 0;
          g->cluster_weight_[i] = 0.0;

          if (g->owner(target_cluster) == me)
          {
              g->cluster_[g->global_to_local(target_cluster)] = cluster_id;
              g->cluster_degree[g->global_to_local(target_cluster)] += g->cluster_degree_[i];
              g->cluster_weight_[g->global_to_local(target_cluster)] += g->cluster_weight_[i];
          }
          else
          {

          }
      }
  }

  return mod;
}

GraphWeight modularity(Graph *g)
{
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(shared), shared(clusterWeight, localCinfo), \
  reduction(+: le_xx), reduction(+: la2_x) schedule(runtime)
#else
#pragma omp parallel for default(shared), shared(clusterWeight, localCinfo), \
  reduction(+: le_xx), reduction(+: la2_x) schedule(static)
#endif
  for (GraphElem i = 0L; i < nv; i++) {
    le_xx += clusterWeight[i];
    la2_x += static_cast<GraphWeight>(localCinfo[i].degree) * static_cast<GraphWeight>(localCinfo[i].degree); 
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

    return mod;
}

private:
};
