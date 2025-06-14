#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unordered_map>
#include <random>
#include <ctime>
#include <iostream>
#include <cstring>
#include <chrono>         
#include "absl/container/flat_hash_map.h"
#define MMULT_N 5


#define CHECK_ALLOC(p)                                                         \
{                                                                              \
    if (!(p)) {                                                                \
        printf("Out of Host memory!");                                         \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_RESULT(r)                                                        \
{                                                                              \
    if (r) {                                                                   \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

using absl::flat_hash_map;

void spmv(uint64_t* edge_start, uint64_t* edge_end, uint64_t* edge_weight, uint64_t* w, uint64_t* z, uint64_t edge_n);
void randomize(uint64_t* v, uint64_t n);
int is_equivalent(uint64_t* w, uint64_t* z, uint64_t* z_c, uint64_t node_n, uint64_t* new_node_n);
int read_file_graph(uint64_t** edge_start, uint64_t** edge_end, uint64_t** edge_weight, uint64_t* edge_n, uint64_t* node_n);
uint64_t read_file_uint64(FILE *file);

int main(void) {
    uint64_t node_n = 0, new_node_n = 0;
    uint64_t *edge_start, *edge_end, *edge_weight, *swp, *w, *z, *z_c, edge_n = 0;

    CHECK_RESULT( read_file_graph(&edge_start, &edge_end, &edge_weight, &edge_n, &node_n) );
    CHECK_ALLOC( w = (uint64_t*)malloc(sizeof(uint64_t) * node_n) );
    CHECK_ALLOC( z = (uint64_t*)malloc(sizeof(uint64_t) * node_n) );
    CHECK_ALLOC( z_c = (uint64_t*)malloc(sizeof(uint64_t) * node_n) );

    for(uint64_t i = 0; i<node_n; ++i){
        w[i] = 1;
        z[i] = 0;
    }

    while(1) {
        for(int i=0; i<MMULT_N;++i) {
            spmv(edge_start, edge_end, edge_weight, w, z, edge_n);
            swp = w;
            w = z;
            z = swp;
            randomize(w, node_n);
        }
        if(is_equivalent(w, z, z_c, node_n, &new_node_n))
            break;
    }

    uint64_t new_edge_n = 0;
    uint64_t first_edge_i = 0;
    uint64_t current_node = edge_start[0];
    unsigned char counting = 1;

    for(uint64_t i = 0; i<new_node_n; ++i) { 
        z[i] = 0;
        w[i] = edge_n;
    }

    for(uint64_t i=0; i<edge_n; ++i) {
        if(i && edge_start[i] != current_node) {
            current_node = edge_start[i];
            counting = !z[z_c[edge_start[i]]];
            first_edge_i = new_edge_n;
            z[z_c[edge_start[i]]] = 1;
        }
        
        if(counting) {
            edge_start[i] = z_c[edge_start[i]];
            edge_end[i] = z_c[edge_end[i]];

            if(w[edge_end[i]] < first_edge_i || w[edge_end[i]] == edge_n) {
                edge_start[new_edge_n] = edge_start[i];
                edge_end[new_edge_n] = edge_end[i];
                edge_weight[new_edge_n] = edge_weight[i];
                w[edge_end[i]] = new_edge_n;
                ++new_edge_n;
            } else {
                edge_weight[w[edge_end[i]]] += edge_weight[i];
            }
        }
    }

    printf("%u\n", new_node_n);
    printf("%lu\n", new_edge_n);

    for(uint64_t i = 0; i< new_edge_n; ++i) {
        printf("%lu %lu %lu\n", edge_start[i], edge_weight[i], edge_end[i]);
    }

    return 0;
}

void spmv(uint64_t* edge_start, uint64_t* edge_end, uint64_t* edge_weight, uint64_t* w, uint64_t* z, uint64_t edge_n) {
    for(uint64_t i=0; i<edge_n; ++i) {
        if(!i || edge_start[i] != edge_start[i-1]) {
            z[edge_start[i]] = 0;
        }
        z[edge_start[i]] += w[edge_end[i]] * edge_weight[i];
    }
}

void randomize(uint64_t* v, uint64_t n) {
    for(uint64_t i=0; i<n; ++i) {
        uint64_t z = v[i] + 0x9e3779b97f4a7c15;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        z = (z ^ (z >> 31)) * 5;
        v[i] = ((z << 7) | (z >> (64 - 7))) * 9;
    }
}

int is_equivalent(uint64_t* w, uint64_t* z, uint64_t* z_c, uint64_t node_n, uint64_t* new_node_n) {
    flat_hash_map<uint64_t, uint64_t> w_unordered_map;
    uint64_t w_last = 1;
    flat_hash_map<uint64_t, uint64_t> z_unordered_map;
    uint64_t z_last = 1;
    uint64_t z_c_cur = 0;
    w_unordered_map.reserve(node_n);
    z_unordered_map.reserve(node_n);
    for(uint64_t i=0; i<node_n; ++i) {
        uint64_t w_val = w[i];
        if(!w_unordered_map[w_val]) {
            w_unordered_map[w_val] = w_last;
            ++w_last;
        }
        uint64_t z_val = z[i];
        if(!(z_c_cur = z_unordered_map[z_val])) {
            z_unordered_map[z_val] = z_last;
            z_c_cur = z_last;
            ++z_last;
        }  
        z_c[i] = z_c_cur - 1;

        if(w_last != z_last) {
            return 0;
        }
    }
    *new_node_n = z_last - 1;
    return 1;
}

int read_file_graph(uint64_t** edge_start, uint64_t** edge_end, uint64_t** edge_weight, uint64_t* edge_n, uint64_t* node_n) {
    FILE *file = fopen("graph.txt", "r");
    *node_n = read_file_uint64(file);
    *edge_n = read_file_uint64(file);
    CHECK_ALLOC( *edge_start = (uint64_t*)malloc(*edge_n * sizeof(uint64_t)) );
    CHECK_ALLOC( *edge_end = (uint64_t*)malloc(*edge_n * sizeof(uint64_t)) );
    CHECK_ALLOC( *edge_weight = (uint64_t*)malloc(*edge_n * sizeof(uint64_t)) );
    for(uint64_t i=0; i<*edge_n; ++i) {
        (*edge_start)[i] = read_file_uint64(file); 
        (*edge_weight)[i] = read_file_uint64(file); 
        (*edge_end)[i] = read_file_uint64(file); 
    }
    return 0;
}

uint64_t read_file_uint64(FILE *file) {
    char ch = fgetc(file);
    uint64_t n = 0;
    uint64_t c = 0;
    while(ch != ' ' && ch != '\n') {
        c = ch - '0';   
        n = (n*10) + c;
        ch = fgetc(file);
    }
    return n;
}
