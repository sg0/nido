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

        Clustering (Graph* g): 
            g_(g), targets_(0), sendbuf_(0),
            nghosts_in_target_(0), nghosts_target_indices_(0), 
            pindex_(0), degree_(-1), prcounts_(0), scounts_(0), 
            rcounts_(0), rdispls_(0), nwin_(MPI_WIN_NULL), 
            winbuf_(nullptr)
    {
        comm_ = g_->get_comm();
        MPI_Comm_size(comm_, &size_);
        MPI_Comm_rank(comm_, &rank_);

        // neighborhood communicator
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
                            == targets_.end())
                    {
                        targets_.push_back(owner);
                    }
                }
            }
        }

        degree_ = targets_.size();
        MPI_Dist_graph_create_adjacent(comm_, targets_.size(), targets_.data(), 
                MPI_UNWEIGHTED, targets_.size(), targets_.data(), MPI_UNWEIGHTED, 
                MPI_INFO_NULL, 0 /*reorder ranks*/, &nbcomm_);

        for (int i = 0; i < degree_; i++)
            pindex_.insert({targets_[i], (GraphElem)i}); 

        nghosts_in_target_.resize(degree_);
        nghosts_target_indices_.resize(degree_);          
        rdispls_.resize(degree_);

        // populate counter that tracks number
        // of ghosts not owned by me
        GraphElem tot_ghosts = 0;

        for (GraphElem i = 0; i < lnv; i++)
        {
            GraphElem e0, e1;
            g_->edge_range(i, e0, e1);

            for (GraphElem e = e0; e < e1; e++)
            {
                Edge const& edge = g_->get_edge(e);
                const int target = g_->get_owner(edge.tail_);

                if (target != rank_)
                {
                    nghosts_in_target_[pindex_[target]] += 1;
                    tot_ghosts += 1;
                }
            }                
        }

        // initialize input buffer

        // sends a pair of vertices with tag,
        // can send at most 2 messages along a
        // cross edge
        sendbuf_ = new GraphElem[tot_ghosts*3*2];

        // allocate MPI windows and lock them

        // TODO FIXME make the same changes for
        // the base RMA version
        MPI_Info info = MPI_INFO_NULL;

#if defined(USE_MPI_ACCUMULATE)
        MPI_Info_create(&info);
        MPI_Info_set(info, "accumulate_ordering", "none");
        MPI_Info_set(info, "accumulate_ops", "same_op");
#endif

        // TODO FIXME report bug, program crashes when g_comm_ used
        MPI_Win_allocate((tot_ghosts*3*2)*sizeof(GraphElem), 
                sizeof(GraphElem), info, comm_, &winbuf_, &nwin_);             

        MPI_Win_lock_all(MPI_MODE_NOCHECK, nwin_);

        // exclusive scan to compute remote 
        // displacements for RMA CALLS
        GraphElem disp = 0;
        for (int t = 0; t < degree_; t++)
        {
            nghosts_target_indices_[t] = disp;
            disp += nghosts_in_target_[t]*3*2;
        }

        // incoming updated prefix sums
        MPI_Neighbor_alltoall(nghosts_target_indices_.data(), 1, 
                MPI_GRAPH_TYPE, rdispls_.data(), 1, MPI_GRAPH_TYPE, g_comm_);

        // set neighbor alltoall params
        scounts_.resize(outdegree_, 0);
        rcounts_.resize(indegree_, 0);
        prcounts_.resize(indegree_, 0);
    }

        ~Clustering() {}

        void clear()
        {
            targets_.clear();

            scounts_.clear();
            rcounts_.clear();
            prcounts_.clear();
            rdispls_.clear();

            nghosts_in_target_.clear();
            nghosts_target_indices_.clear();
            pindex_.clear();

            delete []sendbuf_;

            MPI_Win_unlock_all(nwin_);
            MPI_Win_free(&nwin_);

            MPI_Comm_free(&nbcomm_);
        }

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
        Graph* g_;
        
        std::unordered_map<int, GraphElem> pindex_;                    
        GraphElem* sendbuf_;

        std::vector<int> targets_;
        std::vector<GraphElem> scounts_, rcounts_, prcounts_;
        int indegree_, outdegree_;

        MPI_Win nwin_; 
        GraphElem* winbuf_;
        std::vector<GraphElem> rdispls_, // target displacement
            nghosts_in_target_,          // ghost vertices in target rank
            nghosts_target_indices_;     // indices of data 

        int rank_, size_;
        MPI_Comm comm_;
        MPI_Comm nbcomm_; 
};
