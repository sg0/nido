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
    int origin_;
    GraphElem vid_;
    GraphElem degree_;
    GraphWeight weight_;
    
    Cluster(): origin_(-1), vid_(0), degree_(0), weight_(0.0) {}
    Cluster(int origin, GraphElem vid, GraphElem degree, GraphWeight weight): 
        origin_(origin), vid_(vid), degree_(degree), weight_(weight) 
    {}
};

class Clustering
{
    public:

        Clustering (Graph* g): 
            g_(g), targets_(0), sendbuf_(0), nbcomm_(MPI_COMM_NULL),
            sum_weights_(0.0), constant_term_(0.0), nghosts_in_target_(0), 
            nghosts_target_indices_(0), pindex_(0), degree_(-1), scounts_(0), rcounts_(0), rdispls_(0), 
            nwin_(MPI_WIN_NULL), winbuf_(nullptr), cwin_(MPI_WIN_NULL), cwinbuf_(nullptr), 
            nbr_cluster_degree_(0), nbr_cluster_weight_(0.0)
        {
            comm_ = g_->get_comm();
            MPI_Comm_size(comm_, &size_);
            MPI_Comm_rank(comm_, &rank_);

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
            nbr_cluster_degree_.resize(degree_, 0);
            nbr_cluster_weight_.resize(degree_, 0.0);
 
            MPI_Info info = MPI_INFO_NULL;

#if defined(USE_MPI_ACCUMULATE)
            MPI_Info_create(&info);
            MPI_Info_set(info, "accumulate_ordering", "none");
            MPI_Info_set(info, "accumulate_ops", "same_op");
#endif           

            // window for storing cluster IDs
            MPI_Win_allocate(lnv*sizeof(GraphElem), 
                    sizeof(GraphElem), info, comm_, &cwinbuf_, &cwin_);             
            MPI_Win_lock_all(MPI_MODE_NOCHECK, cwin_);

            // populate counter that tracks number
            // of ghosts not owned by me
            GraphElem tot_ghosts = 0;
            GraphWeight weight_sum = 0.0;

            for (GraphElem i = 0; i < lnv; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);
                
                // initialize cluster id window
                cwinbuf_[i] = g_->local_to_global(i);

                for (GraphElem e = e0; e < e1; e++)
                {
                    Edge const& edge = g_->get_edge(e);
                    const int target = g_->owner(edge.tail_);

                    if (target != rank_)
                    {
                        nghosts_in_target_[pindex_[target]] += 1;
                        tot_ghosts += 1;
                    }

                    weight_sum += edge.weight_;
                }                
            }

            MPI_Allreduce(&weight_sum, &sum_weights_, 1, MPI_WEIGHT_TYPE, MPI_SUM, comm_);
            constant_term_ = 1.0 / sum_weights;
            
            Cluster clust;
            MPI_Aint begin, origin, vid, degree, weight;

            MPI_Get_address(&clust, &begin);
            MPI_Get_address(&clust.origin_, &origin);
            MPI_Get_address(&clust.vid_, &vid);
            MPI_Get_address(&clust.degree_, &degree);
            MPI_Get_address(&clust.weight_, &weight);

            int blens[] = { 1, 1, 1, 1 };
            MPI_Aint displ[] = { origin - begin, vid - begin, degree - begin, weight - begin };
            MPI_Datatype types[] = { MPI_INT, MPI_GRAPH_TYPE, MPI_GRAPH_TYPE, MPI_WEIGHT_TYPE };

            MPI_Type_create_struct(4, blens, displ, types, &mpi_cluster_t_);
            MPI_Type_commit(&mpi_cluster_t_);
            
            // initialize input buffer
            sendbuf_ = new Cluster[tot_ghosts];
            
            // window for storing cluster degrees and weights
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
            nbr_cluster_degree_.clear();
            nbr_cluster_weight_.clear();

            delete []sendbuf_;

            MPI_Win_unlock_all(nwin_);
            MPI_Win_free(&nwin_);
            MPI_Win_unlock_all(cwin_);
            MPI_Win_free(&cwin_);
            
            MPI_Comm_free(&nbcomm_);
            MPI_Type_free(&mpi_cluster_t_);
        }

        GraphWeight get_sum_weights() const 
        { return sum_weights_; }

        GraphWeight constant_term() const
        { return constant_term_; }

        void update_nbr_clusters(MPI_Request *req[2])
        {
            const GraphElem lnv = g_->get_lnv();
            std::vector<int> targets;
            std::vector<GraphElem> out_nbr_cluster_degree(degree_);
            std::vector<GraphWeight> out_nbr_cluster_weight(degree_);
            
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
                        if (std::find(targets.begin(), targets.end(), target) 
                                == targets.end())
                        {
                            targets.push_back(target);
                            out_nbr_cluster_degree[pindex_[target]] = g_->cluster_degree_[i];
                            out_nbr_cluster_weight[pindex_[target]] = g_->cluster_weight_[i];
                        }
                    }
                }                
            }

            // neighbor's cluster degrees and weights
            MPI_Ineighbor_alltoall(out_nbr_cluster_degree.data(), 1, 
                    MPI_GRAPH_TYPE, nbr_cluster_degree_.data(), 1, 
                    MPI_GRAPH_TYPE, nbcomm_, req[0]);
            MPI_Ineighbor_alltoall(out_nbr_cluster_weight.data(), 1, 
                    MPI_WEIGHT_TYPE, nbr_cluster_weight_.data(), 1, 
                    MPI_WEIGHT_TYPE, nbcomm_, req[1]);
        }


        int run_louvain(GraphWeight& iter_mod, GraphWeight lower = -1.0, GraphWeight thresh = 1.0E-06)
        {
            GraphWeight prev_mod = lower;
            GraphWeight mod = -1.0;
            int iters = 0;
            const GraphElem lnv = g_->get_lnv();

            while(true) 
            {
                MPI_Request req[2] = {MPI_REQUEST_NULL};
                
                MPI_Win_flush_all(nwin_);

                rcounts_.clear();
                MPI_Neighbor_alltoall(scounts_.data(), 1, MPI_GRAPH_TYPE, 
                        rcounts_.data(), 1, MPI_GRAPH_TYPE, nbcomm_);
                
                // to store incoming cluster info
                std::vector<std::array<GraphElem, 3>> origin_clusters; 
                
                // fetch cluster info and update local clusters
                for (int k = 0; k < degree_; k++)
                {
                    const GraphElem index = nghosts_target_indices_[k];

                    for (GraphElem i = 0; i < rcounts_[k]; i++)
                    {
                        Cluster clust = winbuf_[index + i];
                        GraphElem target_cluster = -1;

                        // finding a local cluster for an incoming vertex  
                        for (GraphElem v = 0; v < lnv; v++)
                        {
                            GraphElem e0, e1;
                            g_->edge_range(v, e0, e1);
                            eix = g_->cluster_weight_[v];
                            ax = g_->cluster_degree_[v];

                            for (GraphElem e = e0; e < e1; e++)
                            {
                                Edge const& edge = g_->get_edge(e);
                                const int target = g_->owner(edge.tail_);

                                if (target == rank_)
                                {
                                    const GraphElem local_index = g_->global_to_local(edge.tail_);
                                    eiy = g_->cluster_weight_[local_index];
                                    ay = g_->cluster_degree_[local_index];

                                    curr_gain = 2.0 * (eiy - eix) - 2.0 * sum_weights_ * (ay - ax) * constant_term_;

                                    if (curr_gain >= max_gain) 
                                    {
                                        max_gain = curr_gain;
                                        target_cluster = edge.tail_;
                                    }
                                }
                            }
                        }

                        // update target cluster weight/degree
                        g_->cluster_degree_[target_cluster] += clust.degree_;
                        g_->cluster_weight_[target_cluster] += clust.weight_;

                        // buffer outgoing cluster ID window data
                        origin_clusters.push_back({static_cast<GraphElem>(clust.origin_), 
                                target_cluster, g_->global_to_local(clust.vid_, clust.origin_)});
                    }

                    // puts to the origin for updating cluster IDs
                    for (GraphElem i = 0; i < rcounts_[k]; i++)
                    {
#if defined(USE_MPI_ACCUMULATE)
                        MPI_Accumulate(&origin_clusters[i][1], 1, MPI_GRAPH_TYPE, origin_clusters[i][0], 
                                (MPI_Aint)origin_clusters[i][2], 1, MPI_GRAPH_TYPE, MPI_REPLACE, cwin_);
#else
                        MPI_Put(&origin_clusters[i][1], 1, MPI_GRAPH_TYPE, origin_clusters[i][0], 
                                (MPI_Aint)origin_clusters[i][2], 1, MPI_GRAPH_TYPE, cwin_);
#endif
                    }
                }

                update_nbr_cluster_degree(&req);

                // global modularity calculation
                mod = modularity();
                
                if (mod - prev_mod < thresh)
                    break;

                prev_mod = mod;
                
                if (prev_mod < lower)
                    prev_mod = lower;        

                MPI_Wait(&req, MPI_STATUS_IGNORE);
    
                MPI_Win_flush_all(cwin_);
                
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
            const GraphElem lnv = g_->get_lnv();

            // calculate delta-q and determine target community
            for (GraphElem i = 0; i < lnv; i++)
            {
                GraphElem e0, e1;
                g_->edge_range(i, e0, e1);

                GraphElem target_cluster;

                bool is_shifting = false;
                int target_cluster_rank = MPI_PROC_NULL;

                GraphWeight curr_gain = 0.0, max_gain = 0.0, eix = 0.0, 
                            ax = 0.0, eiy = 0.0, ay = 0.0;

                GraphElem e0, e1;
                g_->edge_range(v, e0, e1);
                eix = g_->cluster_weight_[v];
                ax = g_->cluster_degree_[v];

                for (GraphElem e = e0; e < e1; e++)
                {
                    Edge const& edge = g_->get_edge(e);
                    const int target = g_->owner(edge.tail_);

                    if (target == rank_)
                    {
                        const GraphElem local_index = g_->global_to_local(edge.tail_);
                        eiy = g_->cluster_weight_[local_index];
                        ay = g_->cluster_degree_[local_index];
                    }
                    else
                    {
                        eiy = nbr_cluster_weight_[pindex_[target]];
                        ay = nbr_cluster_degree_[pindex_[target]];
                    }

                    curr_gain = 2.0 * (eiy - eix) - 2.0 * sum_weights_ * (ay - ax) * constant_term_;

                    if (curr_gain >= max_gain) 
                    {
                        max_gain = curr_gain;
                        target_cluster_rank = g_->owner(edge.tail_);

                        if (target_cluster_rank == rank_)
                            target_cluster = edge.tail_;
                        else
                            target_cluster = -1;

                        is_shifting = true;
                    }
                }

                // adjust cluster degree and weight
                if (is_shifting)
                {
                    if (target_cluster == -1) // remote buffer
                    {
                        const int pidx = pindex_[target_cluster_rank];
                        const GraphElem curr_count = scounts_[pidx];
                        const GraphElem index = nghosts_target_indices_[pidx] + curr_count;

                        Cluster clust(rank_, g_->local_to_global(i), g_->cluster_degree_[i], g_->cluster_weight_[i]);
                        sendbuf_[index] = clust; 
                        scounts_[pidx]++;
                    }
                    else // local apply
                    {
                        cwinbuf_[i] = target_cluster;
                        g_->cluster_degree_[g_->global_to_local(target_cluster)] += g_->cluster_degree_[i];
                        g_->cluster_weight_[g_->global_to_local(target_cluster)] += g_->cluster_weight_[i];
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
            GraphWeight tot_weights = get_sum_weights();
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

            GraphWeight mod = std::fabs((e_a_xx[0] * constant_term_) - 
                    (e_a_xx[1] * constant_term_ * constant_term_));

            return mod;
        }

    private:
        Graph* g_;
        
        std::unordered_map<int, GraphElem> pindex_;                    
        Cluster *sendbuf_, *winbuf_;
        GraphElem *cwinbuf_;
        GraphWeight sum_weights_;

        std::vector<int> targets_;
        std::vector<GraphElem> scounts_, rcounts_;
        int degree_;
        std::vector<GraphElem> nbr_cluster_degree_;
        
        MPI_Win nwin_, cwin_; 
        std::vector<GraphElem> rdispls_, // target displacement
            nghosts_in_target_,          // ghost vertices in target rank
            nghosts_target_indices_;     // indices of data 

        int rank_, size_;
        MPI_Comm comm_;
        MPI_Comm nbcomm_;
        MPI_Datatype mpi_cluster_t_;
};
#endif
