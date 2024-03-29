#include <omp.h>
#include "louvain_gpu.hpp"
#include "graph_gpu.hpp"
#include "cuda_wrapper.hpp"
 
void LouvainGPU::run(GraphGPU* graph)
{
    cudaStream_t cuStreams[NGPU][2];
    for(int i = 0; i < NGPU; ++i)
    {
        CudaSetDevice(i);
        CudaCall(cudaStreamCreate(&cuStreams[i][0]));
        CudaCall(cudaStreamCreate(&cuStreams[i][1]));
    }

    double total_start, total_end;
    double loop_time = 0;
    #ifdef MULTIPHASE
    double aggregate_time = 0;
    #endif
    int totalLoops = 0;
    bool done = false;

    Int numPhases = 0;
    double Q_old = 0.;

    total_start = omp_get_wtime();
    while(!done && numPhases < MAX_PHASES)
    {
        std::cout << "PHASE #" << numPhases << ": " << std::endl;
        //std::cout << "----------------------------------------\n";

        graph->singleton_partition();
        #ifdef CHECK
        graph->set_community_ids();
        #endif        
        Float Q = graph->compute_modularity();
        Q_old = Q;
        #ifdef CHECK
        graph->compute_modularity_host();
        #endif

        #ifdef PRINT
        std::cout << "LOOP# \tQ \t\tdQ\n";
        std::cout << "----------------------------------------\n";
        std::cout << 0 << " \t" << Q << " \t" << 0 << std::endl;
        #endif

        Float dQ = MAX_FLOAT;
        Int loops = 0;
        double start, end;
        start = omp_get_wtime();
        while(tol_ < dQ and loops < maxLoops_)
        {
            omp_set_num_threads(NGPU);
            #pragma omp parallel
            {
                int g =  omp_get_thread_num() % NGPU;
                CudaSetDevice(g);
            
                for(int batch = 0; batch < nbatches_; ++batch)
                {
                    graph->louvain_update(batch, g); 
                    CudaDeviceSynchronize();
                    #pragma omp barrier 
                    #ifdef CHECK
                    if(g==0)
                        graph->louvain_update_host(batch);
                    #pragma omp barrier
                    #endif 
                    
                    #pragma omp critical
                    {
                        graph->update_community_weights(batch, g);
                        CudaDeviceSynchronize();
                    }
                    //#pragma omp barrier

                    graph->update_community_ids(batch, g);
                    CudaDeviceSynchronize();
                    #pragma omp barrier
                }
            }
            Float Qtmp = graph->compute_modularity();
            dQ = Qtmp - Q;
            loops++;
            if(dQ < 0)
                graph->restore_community();
            else
                Q = Qtmp; 

            #ifdef PRINT
            std::cout << loops << " \t" << Qtmp << " \t" << dQ << std::endl;
            #endif
        }
        end = omp_get_wtime();

        totalLoops += loops;

        if(loops >= maxLoops_)
            std::cout << "Exceed maximum loop number" << std::endl;
        std::cout << "Final Q: "<< Q << std::endl;
        std::cout << "Time elapse " << end-start << " s" << std::endl;

        loop_time += end-start;

        #ifdef MULTIPHASE
        if(Q-Q_old > tol_phase_)
        {
            double start_agg = omp_get_wtime();
            done = graph->aggregation();
            double end_agg =  omp_get_wtime();
            double diff = end_agg-start_agg;
            std::cout << "Aggregation time: " << diff << " s" << std::endl;
            aggregate_time += diff;
        }
        else
            done = true;
        #else
        done = true;
        #endif
        std::cout << "----------------------------------------\n";
        numPhases++;
    }
    total_end = omp_get_wtime();
    double total_time = (double)(total_end-total_start);
    std::cout << "Total time elapse: " << total_time << " s" << std::endl;
    std::cout << "Total Loops: " << totalLoops << "\n";
    std::cout << "Time per loop: " << loop_time/totalLoops << " s/loop\n";
    std::cout << "Aggregation time: " << total_time-loop_time << " s\n";
    std::cout << "Aggregation time: " << aggregate_time << " s\n";

    for(int i = 0; i < NGPU; ++i)
    {
        CudaSetDevice(i);
        CudaCall(cudaStreamDestroy(cuStreams[i][0]));
        CudaCall(cudaStreamDestroy(cuStreams[i][1]));
    }
}
