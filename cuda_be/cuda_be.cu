#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include "absl/container/flat_hash_map.h"
#define THREAD_N 256
#define MMULT_N 5
#define NODE_T uint32_t
#define NODE_T_MAX UINT32_MAX

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_ALLOC(p)                                                         \
{                                                                              \
    if (!(p)) {                                                                \
        printf("Out of Host memory!\n");                                       \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_WEIGHT(tot, c)                                                   \
{                                                                              \
    if (NODE_T_MAX - tot < c) {                                                \
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

int is_equivalent(uint64_t* w, uint64_t* z, NODE_T* z_c, uint8_t* batch_mask, NODE_T* new_node_n, NODE_T node_n);
int read_graph(NODE_T** edge_start, NODE_T** edge_end, NODE_T** edge_weight, uint64_t* edge_n, NODE_T* node_n);
uint64_t read_uint64();

__global__ void init_partition(NODE_T node_n, uint8_t* batch_mask,  uint64_t* z) { 
    NODE_T i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < node_n) {
        z[i] = batch_mask[i] ? node_n : i;
    }
}

__global__ void set_values(uint64_t edge_n, NODE_T* edge_weight, NODE_T* edge_end, uint64_t* values, uint64_t* result) { 
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < edge_n) {
        uint64_t end = edge_end[i];
        uint64_t weight = edge_weight[i];
        values[i] = result[end] * ((uint64_t)weight);
    }
}

__global__ void set_partition(NODE_T unique_node_count, uint8_t* batch_mask, uint64_t* z, uint64_t *result, NODE_T *unique_nodes) {
    NODE_T i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < unique_node_count) {
        NODE_T node = unique_nodes[i];
        z[node] = batch_mask[node] ? result[i] : node;
    }
}

__global__ void randomize(NODE_T node_n, uint64_t* v) { 
    NODE_T i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < node_n) {
        uint64_t z = v[i] + 0x9e3779b97f4a7c15;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        z = (z ^ (z >> 31)) * 5;
        v[i] = ((z << 7) | (z >> (64 - 7))) * 9;
    }
}

int main(int argc, char* argv[]) {
    uint8_t *batch_mask, *d_batch_mask;

    uint64_t edge_n, new_edge_n, *w, *z, *d_w,
             *d_z, *d_unique_edge_count, *d_swp,
             *d_value_node_buffer,max_batch_edge_n,
             batches, *d_edge_buffer; 

    NODE_T node_n, *edge_start, *edge_end, *z_c,
           *edge_weight, *d_key_node_buffer,
           *d_edge_weight, *d_edge_start,
           *d_edge_end, unique_node_count, new_node_n,
           w_unique_n, z_unique_n, *d_unique_node_count,
           *d_w_unique_n, *d_z_unique_n;

    void *d_temp_storage = nullptr;

    size_t temp_sizes_bytes[4] = {0}, max_temp_sizes_bytes = 0;

    auto reduction_op = cuda::std::plus{};

    CHECK_RESULT( read_graph(&edge_start, &edge_end, &edge_weight, &edge_n, &node_n) );
    max_batch_edge_n = argc == 2 ? atoll(argv[1]) : edge_n; 

    if(max_batch_edge_n < node_n) {
        printf("The max batch edge number must be greater than the node number!");
        return EXIT_FAILURE;
    }

    CHECK_ALLOC( w = (uint64_t*)malloc(sizeof(uint64_t) * node_n) );
    CHECK_ALLOC( z = (uint64_t*)malloc(sizeof(uint64_t) * node_n) );
    CHECK_ALLOC( z_c = (NODE_T*)malloc(sizeof(NODE_T) * node_n) );
    CHECK_ALLOC( batch_mask = (uint8_t*)malloc(sizeof(uint8_t) * node_n) );

    CHECK_CUDA( cudaMalloc((void **)&d_batch_mask, node_n * sizeof(uint8_t)) );
    CHECK_CUDA( cudaMalloc((void **)&d_edge_start, max_batch_edge_n * sizeof(NODE_T)) );
    CHECK_CUDA( cudaMalloc((void **)&d_edge_end, max_batch_edge_n * sizeof(NODE_T)) );
    CHECK_CUDA( cudaMalloc((void **)&d_edge_weight, max_batch_edge_n * sizeof(NODE_T)) );
    CHECK_CUDA( cudaMalloc((void **)&d_edge_buffer, max_batch_edge_n * sizeof(uint64_t)) );
    CHECK_CUDA( cudaMalloc((void **)&d_key_node_buffer, node_n * sizeof(NODE_T)) );
    CHECK_CUDA( cudaMalloc((void **)&d_value_node_buffer, node_n * sizeof(uint64_t)) );
    CHECK_CUDA( cudaMalloc((void **)&d_w, node_n * sizeof(uint64_t)) );
    CHECK_CUDA( cudaMalloc((void **)&d_z, node_n * sizeof(uint64_t)) );

    CHECK_CUDA( cudaMalloc((void **)&d_unique_node_count, sizeof(NODE_T)) );
    CHECK_CUDA( cudaMalloc((void **)&d_unique_edge_count, sizeof(uint64_t)) );
    CHECK_CUDA( cudaMalloc((void **)&d_w_unique_n, sizeof(NODE_T)) );
    CHECK_CUDA( cudaMalloc((void **)&d_z_unique_n, sizeof(NODE_T)) );

    cub::DeviceSelect::Unique(d_temp_storage, temp_sizes_bytes[0], d_edge_start, d_key_node_buffer, d_unique_node_count, max_batch_edge_n);
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_sizes_bytes[1], d_edge_start, d_key_node_buffer, d_edge_buffer, d_value_node_buffer, d_unique_node_count, reduction_op, max_batch_edge_n);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_sizes_bytes[2], d_w, d_value_node_buffer, node_n);
    cub::DeviceSelect::Unique(d_temp_storage, temp_sizes_bytes[3], d_value_node_buffer, d_edge_buffer, d_w_unique_n, node_n);

    max_temp_sizes_bytes = *std::max_element(temp_sizes_bytes, temp_sizes_bytes + 4);
    CHECK_CUDA( cudaMalloc(&d_temp_storage, max_temp_sizes_bytes) );

    new_node_n = node_n;
    new_edge_n = edge_n;
    do {
        edge_n = new_edge_n;
        node_n = new_node_n;
        new_node_n = 0;
        for(NODE_T i=0; i<node_n; ++i) z_c[i] = node_n;
        batches = ceil(edge_n / (float)max_batch_edge_n);

        for(uint64_t batch = 0; batch<batches; ++batch) {  
            uint64_t batch_start = max_batch_edge_n * batch;
            uint64_t batch_end = min(edge_n, batch_start + max_batch_edge_n);
            uint64_t batch_edge_n = batch_end - batch_start;

            for(NODE_T i=0; i<node_n; ++i) {
                batch_mask[i] = !batch ? 1 : 0;
            }

            for(uint64_t i=batch_start; i<batch_end; ++i) {
                batch_mask[edge_start[i]] = 1;
            }

            for(uint64_t i=0; i<batch_start; ++i) {
                batch_mask[edge_start[i]] = 0;
            }

            for(uint64_t i=batch_end; i<edge_n; ++i) {
                batch_mask[edge_start[i]] = 0;
            }

            CHECK_CUDA( cudaMemcpy(d_edge_start, edge_start + batch_start, batch_edge_n * sizeof(NODE_T), cudaMemcpyHostToDevice) );
            CHECK_CUDA( cudaMemcpy(d_edge_end, edge_end + batch_start, batch_edge_n * sizeof(NODE_T), cudaMemcpyHostToDevice) );
            CHECK_CUDA( cudaMemcpy(d_edge_weight, edge_weight + batch_start, batch_edge_n * sizeof(NODE_T), cudaMemcpyHostToDevice) );
            CHECK_CUDA( cudaMemcpy(d_batch_mask, batch_mask, node_n * sizeof(uint8_t), cudaMemcpyHostToDevice) );
            cub::DeviceSelect::Unique(d_temp_storage, max_temp_sizes_bytes, d_edge_start, d_key_node_buffer, d_unique_node_count, batch_edge_n);
            CHECK_CUDA( cudaMemcpy(&unique_node_count, d_unique_node_count, sizeof(NODE_T), cudaMemcpyDeviceToHost) );
            init_partition<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(node_n, d_batch_mask, d_w);

            while(1) {
                while(1) {
                    for(int i=0; i < MMULT_N; ++i) {
                        init_partition<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(node_n, d_batch_mask, d_z);

                        set_values<<<(batch_edge_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(batch_edge_n, d_edge_weight, d_edge_end, d_edge_buffer, d_w);

                        cub::DeviceReduce::ReduceByKey(
                          d_temp_storage, max_temp_sizes_bytes,
                          d_edge_start, d_key_node_buffer, d_edge_buffer,
                          d_value_node_buffer, d_unique_node_count, reduction_op, batch_edge_n);

                        set_partition<<<(unique_node_count+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(unique_node_count, d_batch_mask, d_z, d_value_node_buffer, d_key_node_buffer); 
                        randomize<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(node_n, d_z);

                        d_swp = d_z;
                        d_z = d_w;
                        d_w = d_swp;
                    }

                    cub::DeviceRadixSort::SortKeys(d_temp_storage, max_temp_sizes_bytes, d_w, d_value_node_buffer, node_n);
                    cub::DeviceSelect::Unique(d_temp_storage, max_temp_sizes_bytes, d_value_node_buffer, d_edge_buffer, d_w_unique_n, node_n);

                    cub::DeviceRadixSort::SortKeys(d_temp_storage, max_temp_sizes_bytes, d_z, d_value_node_buffer, node_n);
                    cub::DeviceSelect::Unique(d_temp_storage, max_temp_sizes_bytes, d_value_node_buffer, d_edge_buffer, d_z_unique_n, node_n);

                    CHECK_CUDA( cudaMemcpy(&w_unique_n, d_w_unique_n, sizeof(NODE_T), cudaMemcpyDeviceToHost) );
                    CHECK_CUDA( cudaMemcpy(&z_unique_n, d_z_unique_n, sizeof(NODE_T), cudaMemcpyDeviceToHost) );

                    if(w_unique_n == z_unique_n)
                        break;
                }

                CHECK_CUDA( cudaMemcpy(w, d_w, node_n * sizeof(uint64_t), cudaMemcpyDeviceToHost) );
                CHECK_CUDA( cudaMemcpy(z, d_z, node_n * sizeof(uint64_t), cudaMemcpyDeviceToHost) );

                if(is_equivalent(w, z, z_c, batch_mask, &new_node_n, node_n))
                    break;
            }
        }

        for(NODE_T i=0; i<node_n; ++i) {
            if(z_c[i] == node_n) { 
                z_c[i] = new_node_n++;
            }
            //z and w are reused to memorize already considered node ids for compaction and weight indexes respectively 
            z[i] = 0; 
            w[i] = edge_n;
        }

        new_edge_n = 0;
        uint64_t first_edge_i;
        NODE_T current_node;
        uint8_t counting;

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
    } while(batches > 1 && node_n > new_node_n);
        
    printf("%lu\n", batches == 1);
    printf("%lu\n", new_node_n);
    printf("%lu\n", new_edge_n);

    for(uint64_t i = 0; i< new_edge_n; ++i) {
        printf("%lu %lu %lu\n", edge_start[i], edge_weight[i], edge_end[i]);
    }
    return 0;
}

int is_equivalent(uint64_t* w, uint64_t* z, NODE_T* z_c, uint8_t* batch_mask, NODE_T* new_node_n, NODE_T node_n) {
    flat_hash_map<uint64_t, NODE_T> w_unordered_map;
    NODE_T w_seen = 1;
    flat_hash_map<uint64_t, NODE_T> z_unordered_map;
    NODE_T z_seen = 1;
    w_unordered_map.reserve(node_n);
    z_unordered_map.reserve(node_n);
    NODE_T z_current_c;
    NODE_T start_new_node_n = *new_node_n;

    for(NODE_T i=0; i<node_n; ++i) {
        uint64_t w_val = w[i];
        if(!w_unordered_map[w_val]) {
            w_unordered_map[w_val] = w_seen;
            ++w_seen;
        }

        uint64_t z_val = z[i];
        if(!(z_current_c = z_unordered_map[z_val])) {
            if(!batch_mask[i]) {
                z_unordered_map[z_val] = z_seen;
                z_current_c = z_seen;
            } else {
                z_current_c = ++(*new_node_n);
                z_unordered_map[z_val] = z_current_c;
            }
            ++z_seen;
        }
        if(batch_mask[i]) {
            z_c[i] = z_current_c - 1;
        }

        if(w_seen != z_seen) {
            *new_node_n = start_new_node_n;
            return 0;
        }
    }
    return 1;
}

int read_graph(NODE_T** edge_start, NODE_T** edge_end, NODE_T** edge_weight, uint64_t* edge_n, NODE_T* node_n) {
    *node_n = read_uint64();
    *edge_n = read_uint64();
    CHECK_ALLOC( *edge_start = (NODE_T*)malloc(*edge_n * sizeof(NODE_T)) );
    CHECK_ALLOC( *edge_end = (NODE_T*)malloc(*edge_n * sizeof(NODE_T)) );
    CHECK_ALLOC( *edge_weight = (NODE_T*)malloc(*edge_n * sizeof(NODE_T)) );
    NODE_T tot_weight = 0;
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
