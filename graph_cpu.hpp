#ifndef GRAPH_CPU_HPP_
#define GRAPH_CPU_HPP_
#include "types.hpp"
void build_new_vertex_id_cpu
(
    GraphElem* commIdsHost,
    GraphElem* vertexOffsets,
    GraphElem* newNv,
    GraphElem* vertexIds,
    const GraphElem& nv
);
#endif
