#include "louvain_gpu.hpp"
#include "graph_gpu.hpp"
#include "cuda_wrapper.hpp"
 
void LouvainGPU::run(GraphGPU* graph)
{
    cudaStream_t cuStreams[2];
    CudaCall(cudaStreamCreate(&cuStreams[0]));
    CudaCall(cudaStreamCreate(&cuStreams[1]));

    
    cudaEvent_t start, stop;
    cudaEvent_t totalStart, totalStop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventCreate(&totalStart);
    cudaEventCreate(&totalStop);

    bool done = false;
    Int numPhases = 0;
    cudaEventRecord(totalStart, 0);
    while(!done)
    { 
        std::cout << "PHASE #" << numPhases << ": " << std::endl;
        //std::cout << "----------------------------------------\n";

        graph->singleton_partition();

        GraphElem* vertex_partition = graph->get_vertex_partition();
        Int num_partitions = graph->get_num_partitions();
        Float Q = graph->compute_modularity();

        std::cout << "LOOP# \tQ \t\tdQ\n";
        std::cout << "----------------------------------------\n";
        std::cout << 0 << " \t" << Q << " \t" << 0 << std::endl;

        Float dQ = MAX_FLOAT;
        Int loops = 0;
        cudaEventRecord(start, 0);
        while(tol_ < dQ and loops < maxLoops_)
        {
            for(Int part = 0; part < num_partitions; ++part)
            {
                GraphElem v0 = vertex_partition[part+0];
                GraphElem v1 = vertex_partition[part+1];

                GraphElem e0 = graph->get_edge_partition(v0);
                GraphElem e1 = graph->get_edge_partition(v1);

                graph->move_edges_to_device(e0, e1, cuStreams[0]);
                graph->move_weights_to_device(e0, e1, cuStreams[1]);

                GraphElem nv_per_batch = (v1-v0+nbatches_-1)/nbatches_;
                 
                CudaDeviceSynchronize();
            
                for(Int b = 0; b < nbatches_; ++b)
                {
                    GraphElem u0 = b*nv_per_batch+v0;
                    GraphElem u1 = u0 + nv_per_batch;
                    if(u1 > v1) u1 = v1;
                    if(u1 > u0)
                    {
                        GraphElem f0 = graph->get_edge_partition(u0);
                        GraphElem f1 = graph->get_edge_partition(u1);

                        GraphElem f0_local = f0 - e0;
                        //std::cout << u0 << " " << u1 << " " << f0 << " " << f1 << std::endl; 
                        graph->sort_edges_by_community_ids(u0, u1, f0, f1, f0_local); 
                        graph->louvain_update(u0, u1, f0, f1, f0_local);
                    }
                }
                //CudaDeviceSynchronize();
            }
            graph->update_community_ids();
            Float Qtmp = graph->compute_modularity();
            dQ = Qtmp - Q;
            loops++;
            if(dQ < 0)
                graph->update_community_ids();
            else
                Q = Qtmp; 
            std::cout << loops << " \t" << Qtmp << " \t" << dQ << std::endl;
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);

        if(loops >= maxLoops_)
            std::cout << "Exceed maximum loop number" << std::endl;
        std::cout << "Final Q: "<< Q << std::endl;
        std::cout << "Time elapse " << time*1E-03 << " s" << std::endl;
        std::cout << "----------------------------------------\n";

        #ifdef MULTIPHASE
        //CudaDeviceSynchronize();
        done = graph->aggregation();
        #else
        done = true;
        #endif
        numPhases++;
        //CudaDeviceSynchronize();
    }

    cudaEventRecord(totalStop, 0);
    cudaEventSynchronize(totalStop);
    float ttime;
    cudaEventElapsedTime(&ttime, totalStart, totalStop);

    std::cout << "Total time elapse " << ttime*1E-03 << " s" << std::endl;
    //std::cout << "----------------------------------------\n";

    CudaCall(cudaStreamDestroy(cuStreams[0]));
    CudaCall(cudaStreamDestroy(cuStreams[1]));
}
