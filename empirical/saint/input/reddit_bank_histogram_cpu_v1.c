 
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <limits.h>

int reddit_bank_histogram_abi_version(void) { return 1; }

static int count_int32(const int32_t *ids, int64_t length, int32_t *counts,
                       int64_t population, int32_t limit) {
    for (int64_t i = 0; i < length; ++i) {
        const int32_t value = ids[i];
        if (value < 0) return 3;
        if ((int64_t)value >= population) return 4;
        const int32_t next = counts[value] + 1;
        if (next > limit) return 5;
        counts[value] = next;
    }
    return 0;
}

int reddit_bank_histogram_count(
    const int32_t *node_ids, int64_t node_length,
    const int32_t *edge_ids, int64_t edge_length,
    double *node_output, int64_t node_population,
    float *edge_output, int64_t edge_population,
    int32_t graph_count) {
    if (node_length < 0 || edge_length < 0 || node_population < 0 || edge_population < 0
        || graph_count != 1000
        || (uint64_t)node_population > SIZE_MAX / sizeof(int32_t)
        || (uint64_t)edge_population > SIZE_MAX / sizeof(int32_t)) return 1;
    if ((node_length && !node_ids) || (edge_length && !edge_ids)
        || (node_population && !node_output) || (edge_population && !edge_output)) return 1;
    int32_t *nodes = node_population ? calloc((size_t)node_population, sizeof(int32_t)) : NULL;
    int32_t *edges = edge_population ? calloc((size_t)edge_population, sizeof(int32_t)) : NULL;
    if ((node_population && !nodes) || (edge_population && !edges)) {
        free(nodes); free(edges); return 2;
    }
    int status = count_int32(node_ids, node_length, nodes, node_population, graph_count);
    if (!status) status = count_int32(edge_ids, edge_length, edges, edge_population, graph_count);
     
    if (!status) {
        for (int64_t i = 0; i < node_population; ++i) node_output[i] = (double)nodes[i];
        for (int64_t i = 0; i < edge_population; ++i) edge_output[i] = (float)edges[i];
    }
    free(nodes); free(edges);
    return status;
}
