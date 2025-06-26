#include <stdlib.h>
#include <stdio.h>
#include <chrono>         
#include "absl/container/flat_hash_map.h"
#define MMULT_N 5
#define WEIGHT_MAX UINT32_MAX

#define CHECK_ALLOC(p)                                                         \
{                                                                              \
    if (!(p)) {                                                                \
        printf("Out of Host memory!");                                         \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_WEIGHT(tot, c)                                                   \
{                                                                              \
    if (WEIGHT_MAX - tot < c) {                                                \
        printf("Total edge weight exceeding limit!\n");                        \
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
typedef uint32_t node_t;

void spmv(node_t* edge_start, node_t* edge_end, node_t* edge_weight, uint64_t* w, uint64_t* z, uint64_t edge_n);
void randomize(uint64_t* v, node_t n);
int is_equivalent(uint64_t* w, uint64_t* z, node_t* z_c, node_t node_n, node_t* new_node_n);
int read_graph(node_t** edge_start, node_t** edge_end, node_t** edge_weight, uint64_t* edge_n, node_t* node_n);
uint64_t read_uint64();

int main(void) {
    node_t node_n = 0, new_node_n = 0, *edge_start, *edge_end, *edge_weight, *z_c;
    uint64_t *swp, *w, *z, edge_n = 0;

    CHECK_RESULT( read_graph(&edge_start, &edge_end, &edge_weight, &edge_n, &node_n) );
    CHECK_ALLOC( w = (uint64_t*)malloc(sizeof(uint64_t) * node_n) );
    CHECK_ALLOC( z = (uint64_t*)malloc(sizeof(uint64_t) * node_n) );
    CHECK_ALLOC( z_c = (node_t*)malloc(sizeof(node_t) * node_n) );

    auto st = std::chrono::steady_clock::now();
    for(node_t i = 0; i<node_n; ++i){
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
    uint64_t first_edge_i;
    node_t current_node;
    uint8_t counting;

    for(node_t i = 0; i<new_node_n; ++i) { 
        z[i] = 0;
        w[i] = edge_n;
    }

    for(uint64_t i=0; i<edge_n; ++i) {
        if(!i || edge_start[i] != current_node) {
            current_node = edge_start[i];
            counting = !z[z_c[edge_start[i]]];
            first_edge_i = new_edge_n;
            z[z_c[edge_start[i]]] = 1;
        }
        
        if(counting) {
            edge_start[i] = z_c[edge_start[i]];
            edge_end[i] = z_c[edge_end[i]];

            if(w[edge_end[i]] == edge_n || w[edge_end[i]] < first_edge_i) {
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

    auto en = std::chrono::steady_clock::now();
    double time_s = std::chrono::duration_cast<std::chrono::microseconds>(en - st).count() / 1000000.0;
    printf("%f\n", time_s);
    printf("%u\n", new_node_n);
    printf("%lu\n", new_edge_n);

    return 0;
}

void spmv(node_t* edge_start, node_t* edge_end, node_t* edge_weight, uint64_t* w, uint64_t* z, uint64_t edge_n) {
    for(uint64_t i=0; i<edge_n; ++i) {
        if(!i || edge_start[i] != edge_start[i-1]) {
            z[edge_start[i]] = 0;
        }
        z[edge_start[i]] += w[edge_end[i]] * edge_weight[i];
    }
}

void randomize(uint64_t* v, node_t n) {
    for(node_t i=0; i<n; ++i) {
        uint64_t z = v[i] + 0x9e3779b97f4a7c15;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        z = (z ^ (z >> 31)) * 5;
        v[i] = ((z << 7) | (z >> (64 - 7))) * 9;
    }
}

int is_equivalent(uint64_t* w, uint64_t* z, node_t* z_c, node_t node_n, node_t* new_node_n) {
    flat_hash_map<uint64_t, node_t> w_unordered_map;
    node_t w_last = 1;
    flat_hash_map<uint64_t, node_t> z_unordered_map;
    node_t z_last = 1;
    node_t z_c_cur = 0;
    w_unordered_map.reserve(node_n);
    z_unordered_map.reserve(node_n);
    for(node_t i=0; i<node_n; ++i) {
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

int read_graph(node_t** edge_start, node_t** edge_end, node_t** edge_weight, uint64_t* edge_n, node_t* node_n) {
    *node_n = read_uint64();
    *edge_n = read_uint64();
    CHECK_ALLOC( *edge_start = (node_t*)malloc(*edge_n * sizeof(node_t)) );
    CHECK_ALLOC( *edge_end = (node_t*)malloc(*edge_n * sizeof(node_t)) );
    CHECK_ALLOC( *edge_weight = (node_t*)malloc(*edge_n * sizeof(node_t)) );
    node_t tot_weight = 0;
    for(uint64_t i=0; i<*edge_n; ++i) {
        (*edge_start)[i] = read_uint64(); 
        (*edge_weight)[i] = read_uint64(); 
        CHECK_WEIGHT(tot_weight, (*edge_weight)[i]);
        tot_weight += (*edge_weight)[i];
        (*edge_end)[i] = read_uint64(); 
    }
    return 0;
}

uint64_t read_uint64() {
    char ch = getchar();
    uint64_t n = 0;
    uint64_t c = 0;
    while(ch != ' ' && ch != '\n') {
        c = ch - '0';   
        n = (n*10) + c;
        ch = getchar();
    }
    return n;
}
