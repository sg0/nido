#pragma once
#ifndef CLUSTERING_HPP
#define CLUSTERING_HPP

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

struct Cluster
{
    GraphElem vid_;
    GraphElem degree_;
    GraphWeight weight_;
    
    Cluster(): vid_(-1), degree_(0), weight_(0.0) {}
    Cluster(GraphElem vid, GraphElem degree, GraphWeight weight): 
        vid_(vid), degree_(degree), weight_(weight) 
    {}
};

class Clustering
{
    public:

        Clustering (Graph* g): 
            g_(g), targets_(0), sendbuf_(0),
            nghosts_in_target_(0), nghosts_target_indices_(0), 
            pindex_(0), degree_(-1), scounts_(0), rcounts_(0), 
            rdispls_(0), nwin_(MPI_WIN_NULL), winbuf_(nullptr)
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
                    const int owner = g_->owner(edge.tail_); 
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
                    const int target = g_->owner(edge.tail_);

                    if (target != rank_)
                    {
                        nghosts_in_target_[pindex_[target]] += 1;
                        tot_ghosts += 1;
                    }
                }                
            }

            Cluster clust;
            MPI_Aint begin, id, degree, weight;

            MPI_Get_address(&clust, &begin);
            MPI_Get_address(&clust.vid_, &id);
            MPI_Get_address(&clust.degree_, &degree);
            MPI_Get_address(&clust.weight_, &weight);

            int blens[] = { 1, 1, 1 };
            MPI_Aint displ[] = { id - begin, degree - begin, weight - begin };
            MPI_Datatype types[] = { MPI_GRAPH_TYPE, MPI_GRAPH_TYPE, MPI_WEIGHT_TYPE };

            MPI_Type_create_struct(3, blens, displ, types, &mpi_cluster_t_);
            MPI_Type_commit(&mpi_cluster_t_);
            
            // initialize input buffer
            sendbuf_ = new Cluster[tot_ghosts];

            MPI_Info info = MPI_INFO_NULL;

#if defined(USE_MPI_ACCUMULATE)
            MPI_Info_create(&info);
            MPI_Info_set(info, "accumulate_ordering", "none");
            MPI_Info_set(info, "accumulate_ops", "same_op");
#endif

            MPI_Win_allocate(tot_ghosts*sizeof(Cluster), 
                    sizeof(Cluster), info, comm_, &winbuf_, &nwin_);             

            MPI_Win_lock_all(MPI_MODE_NOCHECK, nwin_);

            // exclusive scan to compute remote 
            // displacements for RMA CALLS
            GraphElem disp = 0;
            for (int t = 0; t < degree_; t++)
            {
                nghosts_target_indices_[t] = disp;
                disp += nghosts_in_target_[t];
            }

            // incoming updated prefix sums
            MPI_Neighbor_alltoall(nghosts_target_indices_.data(), 1, 
                    MPI_GRAPH_TYPE, rdispls_.data(), 1, MPI_GRAPH_TYPE, nbcomm_);

            // set neighbor alltoall params
            scounts_.resize(degree_, 0);
            rcounts_.resize(degree_, 0);
        }

        ~Clustering() {}

        void clear()
        {
            targets_.clear();

            scounts_.clear();
            rcounts_.clear();
            rdispls_.clear();

            nghosts_in_target_.clear();
            nghosts_target_indices_.clear();
            pindex_.clear();

            delete []sendbuf_;

            MPI_Win_unlock_all(nwin_);
            MPI_Win_free(&nwin_);

            MPI_Comm_free(&nbcomm_);
            MPI_Type_free(&mpi_cluster_t_);
        }

        int run_louvain(GraphWeight& iter_mod, GraphWeight lower = -1.0, GraphWeight thresh = 1.0E-06)
        {
            GraphWeight prev_mod = lower;
            GraphWeight mod = -1.0;
            int iters = 0;

            while(true) 
            {
                MPI_Win_flush_all(nwin_);

                rcounts_.clear();
                MPI_Neighbor_alltoall(scounts_.data(), 1, MPI_GRAPH_TYPE, 
                        rcounts_.data(), 1, MPI_GRAPH_TYPE, nbcomm_);

                // fetch cluster info and update local clusters
                for (int k = 0; k < degree_; k++)
                {
                    const GraphElem index = nghosts_target_indices_[k];

                    for (GraphElem i = 0; i < rcounts_[k]; i++)
                    {
                        Cluster clust = winbuf_[index + i];
                        const GraphElem vid = g_->global_to_local(clust.vid_);
                        g_->cluster_degree_[vid] += clust.degree_;
                        g_->cluster_weight_[vid] += clust.weight_;
                    }
                }

                // global modularity calculation
                mod = modularity();
                
                if (mod - prev_mod < thresh)
                    break;

                prev_mod = mod;
                
                if (prev_mod < lower)
                    prev_mod = lower;              
    
                // determine target cluster based on local computation 
                louvain_iteration();

                // push out cluster size/degree to process containing 
                // target clusters for remotely owned vertices
                outstanding_puts();
 
                iters++;

                if (iters >= DEFAULT_LOUVAIN_ITERS)
                    break;
            }

            iter_mod = prev_mod;

            return iters;
        }

        void louvain_iteration()
        {
            GraphWeight le_xx = 0.0, la2_x = 0.0, local_mod, mod;
            const GraphElem lnv = g_->get_lnv();
            GraphWeight tot_weights = g_->get_sum_weights();
            GraphWeight constant_term = 1.0 / (GraphWeight)2.0 * tot_weights;
            
            // local modularity
            for (GraphElem i = 0; i < lnv; i++) 
            {
                le_xx += g_->cluster_weight_[i];
                la2_x += static_cast<GraphWeight>(g_->cluster_degree_[i]) * static_cast<GraphWeight>(g_->cluster_degree_[i]); 
            } 

            mod = std::fabs((le_xx * constant_term) - (la2_x * constant_term * constant_term));

            // calculate delta-q and determine target community
            for (GraphElem i = 0; i < lnv; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);
                GraphElem target_cluster = g_->cluster_[i];
                GraphWeight upd_mod = 0.0;

                for (GraphElem e = e0; e < e1; e++)
                {
                    GraphWeight la2_xup = la2_x; // le_xx will be unchanged
                    Edge const& edge = g_->get_edge(e);

                    if (g_->owner(edge.tail_) == rank_)
                    {
                        GraphElem curr_degree = g_->cluster_degree_[i];
                        curr_degree += g_->cluster_degree_[g_->global_to_local(edge.tail_)];
                        la2_xup += static_cast<GraphWeight>(curr_degree) * static_cast<GraphWeight>(curr_degree) 
                            - static_cast<GraphWeight>(g_->cluster_degree_[i]) * static_cast<GraphWeight>(g_->cluster_degree_[i]);
                    }
                    else
                        la2_xup -= static_cast<GraphWeight>(g_->cluster_degree_[i]) * static_cast<GraphWeight>(g_->cluster_degree_[i]);

                    upd_mod = std::fabs((le_xx * constant_term) - (la2_xup * constant_term * constant_term));

                    if (upd_mod > mod)
                    {
                        mod = upd_mod;
                        target_cluster = edge.tail_;
                    }
                }

                // adjust cluster degree and weight
                if (target_cluster != g_->cluster_[i])
                {
                    g_->cluster_[i] = target_cluster;
                    const int target = g_->owner(target_cluster);
                    if (target == rank_)
                    {
                        g_->cluster_degree_[g_->global_to_local(target_cluster)] += g_->cluster_degree_[i];
                        g_->cluster_weight_[g_->global_to_local(target_cluster)] += g_->cluster_weight_[i];
                    }
                    else
                    {
                        const int pidx = pindex_[target];
                        const GraphElem curr_count = scounts_[pidx];
                        const GraphElem index = nghosts_target_indices_[pidx] + curr_count;

                        Cluster clust(target_cluster, g_->cluster_degree_[i], g_->cluster_weight_[i]);
                        sendbuf_[index] = clust; 
                        scounts_[pidx]++;
                    }
                 
                    g_->cluster_degree_[i] = 0;
                    g_->cluster_weight_[i] = 0.0;
                }
            }
        }

        void outstanding_puts()
        {
            for (int i = 0; i < degree_; i++)
            {
                const GraphElem pidx = pindex_[targets_[i]];
                const GraphElem curr_count = scounts_[pidx];
                const GraphElem index = nghosts_target_indices_[pidx];
                GraphElem tdisp = rdispls_[pidx];

#if defined(USE_MPI_ACCUMULATE)
                MPI_Accumulate(&sendbuf_[index], curr_count, mpi_cluster_t, targets_[i], 
                        (MPI_Aint)tdisp, curr_count, mpi_cluster_t_, MPI_REPLACE, nwin_);
#else
                MPI_Put(&sendbuf_[index], curr_count, mpi_cluster_t_, targets_[i], 
                        (MPI_Aint)tdisp, curr_count, mpi_cluster_t_, nwin_);
#endif
            }
        }

        GraphWeight modularity()
        {
            const GraphElem lnv = g_->get_lnv();
            GraphWeight tot_weights = g_->get_sum_weights();
            GraphWeight constant_term = 1.0 / (GraphWeight)2.0 * tot_weights;
            GraphWeight le_la_xx[2];
            GraphWeight e_a_xx[2] = {0.0, 0.0};
            GraphWeight le_xx = 0.0, la2_x = 0.0;
            GraphElem *cluster_degree = g_->cluster_degree_.data();
            GraphWeight *cluster_weight = g_->cluster_weight_.data();

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(shared), shared(cluster_weight, cluster_degree), \
            reduction(+: le_xx), reduction(+: la2_x) schedule(runtime)
#else
#pragma omp parallel for default(shared), shared(cluster_weight, cluster_degree), \
            reduction(+: le_xx), reduction(+: la2_x) schedule(static)
#endif
            for (GraphElem i = 0; i < lnv; i++) 
            {
                le_xx += cluster_weight[i];
                la2_x += static_cast<GraphWeight>(cluster_degree[i]) * static_cast<GraphWeight>(cluster_degree[i]); 
            } 
            le_la_xx[0] = le_xx;
            le_la_xx[1] = la2_x;

            MPI_Allreduce(le_la_xx, e_a_xx, 2, MPI_WEIGHT_TYPE, MPI_SUM, comm_);

            GraphWeight mod = std::fabs((e_a_xx[0] * constant_term) - 
                    (e_a_xx[1] * constant_term * constant_term));

            return mod;
        }

    private:
        Graph* g_;
        
        std::unordered_map<int, GraphElem> pindex_;                    
        Cluster *sendbuf_, *winbuf_;

        std::vector<int> targets_;
        std::vector<GraphElem> scounts_, rcounts_;
        int degree_;

        MPI_Win nwin_; 
        std::vector<GraphElem> rdispls_, // target displacement
            nghosts_in_target_,          // ghost vertices in target rank
            nghosts_target_indices_;     // indices of data 

        int rank_, size_;
        MPI_Comm comm_;
        MPI_Comm nbcomm_;
        MPI_Datatype mpi_cluster_t_;
};
#endif
