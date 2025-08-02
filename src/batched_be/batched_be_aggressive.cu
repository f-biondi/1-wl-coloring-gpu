#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <omp.h>
#include <chrono>         
#include "absl/container/flat_hash_map.h"
#define THREAD_N 256
#define MMULT_N 5
#define WEIGHT_MAX UINT64_MAX

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUDA_LOG(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
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
typedef uint64_t weight_t;

int is_equivalent(uint64_t* w, uint64_t* z, node_t* z_c, uint8_t* batch_mask, node_t node_n);
int read_graph(node_t** edge_start, node_t** edge_end, weight_t** edge_weight, uint64_t* edge_n, node_t* node_n);
uint64_t read_uint64();

__global__ void init_partition(node_t node_n, uint8_t* batch_mask,  uint64_t* z, int gpu) { 
    node_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < node_n) {
        z[i] = batch_mask[i] ? node_n : i;
    }
}

__global__ void set_values(uint64_t edge_n, weight_t* edge_weight, node_t* edge_end, uint64_t* values, uint64_t* result) { 
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < edge_n) {
        uint64_t end = edge_end[i];
        uint64_t weight = edge_weight[i];
        values[i] = result[end] * ((uint64_t)weight);
    }
}

__global__ void set_partition(node_t unique_node_count, uint8_t* batch_mask, uint64_t* z, uint64_t *result, node_t *unique_nodes) {
    node_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < unique_node_count) {
        node_t node = unique_nodes[i];
        z[node] = batch_mask[node] ? result[i] : node;
    }
}

__global__ void randomize(node_t node_n, uint64_t* v) { 
    node_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < node_n) {
        uint64_t z = v[i] + 0x9e3779b97f4a7c15;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        z = (z ^ (z >> 31)) * 5;
        v[i] = ((z << 7) | (z >> (64 - 7))) * 9;
    }
}

int main(int argc, char* argv[]) {
    uint8_t *batch_mask, **d_batch_mask;

    uint64_t edge_n, new_edge_n, *w, *z,
             **d_w, **d_z, **d_unique_edge_count, **d_swp,
             **d_value_node_buffer, **d_edge_buffer,
             max_batch_edge_n, max_gpus, batches; 

    node_t node_n, start_node_n, *edge_start, *edge_end, *z_c,
           *final_z, **d_key_node_buffer, **d_edge_start,
           **d_edge_end, *unique_node_count, new_node_n,
           *w_unique_n, *z_unique_n, **d_unique_node_count,
           **d_w_unique_n, **d_z_unique_n;

    weight_t *edge_weight, **d_edge_weight;

    void **d_temp_storage;

    size_t* max_temp_sizes_bytes;

    auto reduction_op = cuda::std::plus{};

    CHECK_RESULT( read_graph(&edge_start, &edge_end, &edge_weight, &edge_n, &node_n) );
    max_batch_edge_n = argc >= 2 ? atoll(argv[1]) : edge_n; 
    max_gpus = argc == 3 ? atoll(argv[2]) : 1; 

    CHECK_ALLOC( w = (uint64_t*)malloc(sizeof(uint64_t) * node_n * max_gpus) );
    CHECK_ALLOC( z = (uint64_t*)malloc(sizeof(uint64_t) * node_n * max_gpus) );
    CHECK_ALLOC( z_c = (node_t*)malloc(sizeof(node_t) * node_n) );
    CHECK_ALLOC( final_z  = (node_t*)malloc(sizeof(node_t) * node_n) );
    CHECK_ALLOC( batch_mask = (uint8_t*)malloc(sizeof(uint8_t) * node_n * max_gpus) );
    CHECK_ALLOC( max_temp_sizes_bytes = (size_t*)malloc(sizeof(size_t) * max_gpus) );

    CHECK_ALLOC( d_temp_storage = (void**)malloc(max_gpus * sizeof(void*)) );
    CHECK_ALLOC( d_batch_mask = (uint8_t**)malloc(max_gpus * sizeof(uint8_t*)) );
    CHECK_ALLOC( d_w = (uint64_t**)malloc(max_gpus * sizeof(uint64_t*)) );
    CHECK_ALLOC( d_z = (uint64_t**)malloc(max_gpus * sizeof(uint64_t*)) );
    CHECK_ALLOC( d_unique_edge_count = (uint64_t**)malloc(max_gpus * sizeof(uint64_t*)) );
    CHECK_ALLOC( d_swp = (uint64_t**)malloc(max_gpus * sizeof(uint64_t*)) );
    CHECK_ALLOC( d_value_node_buffer = (uint64_t**)malloc(max_gpus * sizeof(uint64_t*)) );
    CHECK_ALLOC( d_edge_buffer = (uint64_t**)malloc(max_gpus * sizeof(uint64_t*)) );
    CHECK_ALLOC( d_swp = (uint64_t**)malloc(max_gpus * sizeof(uint64_t*)) );

    CHECK_ALLOC( d_key_node_buffer = (node_t**)malloc(max_gpus * sizeof(node_t*)) );
    CHECK_ALLOC( d_edge_weight = (weight_t**)malloc(max_gpus * sizeof(weight_t*)) );
    CHECK_ALLOC( d_edge_start = (node_t**)malloc(max_gpus * sizeof(node_t*)) );
    CHECK_ALLOC( d_edge_end = (node_t**)malloc(max_gpus * sizeof(node_t*)) );
    CHECK_ALLOC( d_unique_node_count = (node_t**)malloc(max_gpus * sizeof(node_t*)) );
    CHECK_ALLOC( d_w_unique_n = (node_t**)malloc(max_gpus * sizeof(node_t*)) );
    CHECK_ALLOC( d_z_unique_n = (node_t**)malloc(max_gpus * sizeof(node_t*)) );

    CHECK_ALLOC( unique_node_count = (node_t*)malloc(max_gpus * sizeof(node_t)) );
    CHECK_ALLOC( w_unique_n = (node_t*)malloc(max_gpus * sizeof(node_t)) );
    CHECK_ALLOC( z_unique_n = (node_t*)malloc(max_gpus * sizeof(node_t)) );

    for(uint64_t i=0; i<max_gpus; ++i) { 
        cudaSetDevice(i);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        size_t temp_sizes_bytes[4] = {0};
        CHECK_CUDA( cudaMalloc((void **)&d_batch_mask[i], node_n * sizeof(uint8_t)) );
        CHECK_CUDA( cudaMalloc((void **)&d_edge_start[i], max_batch_edge_n * sizeof(node_t)) );
        CHECK_CUDA( cudaMalloc((void **)&d_edge_end[i], max_batch_edge_n * sizeof(node_t)) );
        CHECK_CUDA( cudaMalloc((void **)&d_edge_weight[i], max_batch_edge_n * sizeof(weight_t)) );
        CHECK_CUDA( cudaMalloc((void **)&d_edge_buffer[i], max((uint64_t)node_n, max_batch_edge_n) * sizeof(uint64_t)) );
        CHECK_CUDA( cudaMalloc((void **)&d_key_node_buffer[i], node_n * sizeof(node_t)) );
        CHECK_CUDA( cudaMalloc((void **)&d_value_node_buffer[i], node_n * sizeof(uint64_t)) );

        CHECK_CUDA( cudaMalloc((void **)&d_w[i], node_n * sizeof(uint64_t)) );
        CHECK_CUDA( cudaMalloc((void **)&d_z[i], node_n * sizeof(uint64_t)) );

        CHECK_CUDA( cudaMalloc((void **)&d_unique_node_count[i], sizeof(node_t)) );
        CHECK_CUDA( cudaMalloc((void **)&d_unique_edge_count[i], sizeof(uint64_t)) );
        CHECK_CUDA( cudaMalloc((void **)&d_w_unique_n[i], sizeof(node_t)) );
        CHECK_CUDA( cudaMalloc((void **)&d_z_unique_n[i], sizeof(node_t)) );

        cub::DeviceSelect::Unique(nullptr, temp_sizes_bytes[0], d_edge_start[i], d_key_node_buffer[i], d_unique_node_count[i], max_batch_edge_n, stream);
        cub::DeviceReduce::ReduceByKey(nullptr, temp_sizes_bytes[1], d_edge_start[i], d_key_node_buffer[i], d_edge_buffer[i], d_value_node_buffer[i],
                d_unique_node_count[i], reduction_op, max_batch_edge_n, stream);
        cub::DeviceRadixSort::SortKeys(nullptr, temp_sizes_bytes[2], d_w[i], d_value_node_buffer[i], node_n, 0, sizeof(uint64_t) * 8, stream);
        cub::DeviceSelect::Unique(nullptr, temp_sizes_bytes[3], d_value_node_buffer[i], d_edge_buffer[i], d_w_unique_n[i], node_n, stream);

        CHECK_CUDA( cudaStreamSynchronize(stream) );

        max_temp_sizes_bytes[i] = *std::max_element(temp_sizes_bytes, temp_sizes_bytes + 4);
        CHECK_CUDA( cudaMalloc(&d_temp_storage[i], max_temp_sizes_bytes[i]) );
        cudaStreamDestroy(stream);
    }
    cudaDeviceSynchronize();

    for(node_t i=0; i<node_n; ++i) final_z[i] = i;

    auto st = std::chrono::steady_clock::now();

    start_node_n = node_n;
    new_node_n = node_n;
    new_edge_n = edge_n;
    do {
        edge_n = new_edge_n;
        node_n = new_node_n;
        new_node_n = 0;
        for(node_t i=0; i<node_n; ++i) z_c[i] = node_n;
        batches = ceil(edge_n / (float)max_batch_edge_n);

        uint64_t i, gpu,  batch, batch_start, batch_end, batch_edge_n, *current_w, *current_z;
        cudaStream_t stream;
        uint8_t* current_batch_mask;

        #pragma omp parallel num_threads(max_gpus) private(i, stream, gpu, batch, batch_start, batch_end, batch_edge_n, current_batch_mask, current_w, current_z) 
        {
            gpu = omp_get_thread_num();
            CHECK_CUDA_LOG( cudaSetDevice(gpu) );
            CHECK_CUDA_LOG( cudaStreamCreate(&stream) );

            #pragma omp for
            for(batch = 0; batch<batches; ++batch) {  
                batch_start = max_batch_edge_n * batch;
                batch_end = min(edge_n, batch_start + max_batch_edge_n);
                batch_edge_n = batch_end - batch_start;

                current_batch_mask = batch_mask + (node_n * gpu);
                for(i=0; i<node_n; ++i) {
                    current_batch_mask[i] = !batch ? 1 : 0;
                }

                for(i=batch_start; i<batch_end; ++i) {
                    current_batch_mask[edge_start[i]] = 1;
                }

                for(i=0; i<batch_start; ++i) {
                    current_batch_mask[edge_start[i]] = 0;
                }

                for(i=batch_end; i<edge_n; ++i) {
                    current_batch_mask[edge_start[i]] = 0;
                }

                cudaMemcpyAsync(d_edge_start[gpu], edge_start + batch_start, batch_edge_n * sizeof(node_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_edge_end[gpu], edge_end + batch_start, batch_edge_n * sizeof(node_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_edge_weight[gpu], edge_weight + batch_start, batch_edge_n * sizeof(weight_t), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_batch_mask[gpu], current_batch_mask, node_n * sizeof(uint8_t), cudaMemcpyHostToDevice, stream);
                cudaStreamSynchronize(stream);

                cub::DeviceSelect::Unique(
                    d_temp_storage[gpu],
                    max_temp_sizes_bytes[gpu],
                    d_edge_start[gpu],
                    d_key_node_buffer[gpu],
                    d_unique_node_count[gpu],
                    batch_edge_n,
                    stream
                );

                cudaMemcpyAsync(&unique_node_count[gpu], d_unique_node_count[gpu], sizeof(node_t), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);

                init_partition<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N, 0, stream>>>(node_n, d_batch_mask[gpu], d_w[gpu],gpu);

                while(1) {
                    while(1) {
                        for(i=0; i < MMULT_N; ++i) {
                            init_partition<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N, 0, stream>>>(node_n, d_batch_mask[gpu], d_z[gpu],gpu);

                            set_values<<<(batch_edge_n+(THREAD_N-1)) / THREAD_N, THREAD_N, 0, stream>>>(
                                batch_edge_n,
                                d_edge_weight[gpu],
                                d_edge_end[gpu],
                                d_edge_buffer[gpu],
                                d_w[gpu]
                            );

                            cub::DeviceReduce::ReduceByKey(
                                d_temp_storage[gpu],
                                max_temp_sizes_bytes[gpu],
                                d_edge_start[gpu],
                                d_key_node_buffer[gpu],
                                d_edge_buffer[gpu],
                                d_value_node_buffer[gpu],
                                d_unique_node_count[gpu],
                                reduction_op,
                                batch_edge_n,
                                stream
                            );

                            set_partition<<<(unique_node_count[gpu]+(THREAD_N-1)) / THREAD_N, THREAD_N, 0, stream>>>
                            (
                                unique_node_count[gpu],
                                d_batch_mask[gpu],
                                d_z[gpu],
                                d_value_node_buffer[gpu],
                                d_key_node_buffer[gpu]
                            ); 

                            randomize<<<(node_n+(THREAD_N-1)) / THREAD_N, THREAD_N, 0, stream>>>(node_n, d_z[gpu]);

                            d_swp[gpu] = d_z[gpu];
                            d_z[gpu] = d_w[gpu];
                            d_w[gpu] = d_swp[gpu];
                        }

                        cub::DeviceRadixSort::SortKeys(
                            d_temp_storage[gpu],
                            max_temp_sizes_bytes[gpu],
                            d_w[gpu],
                            d_value_node_buffer[gpu],
                            node_n,
                            0,
                            sizeof(uint64_t) * 8,
                            stream
                        );

                        cub::DeviceSelect::Unique(
                            d_temp_storage[gpu],
                            max_temp_sizes_bytes[gpu],
                            d_value_node_buffer[gpu],
                            d_edge_buffer[gpu],
                            d_w_unique_n[gpu],
                            node_n,
                            stream
                        );

                        cub::DeviceRadixSort::SortKeys(
                            d_temp_storage[gpu],
                            max_temp_sizes_bytes[gpu],
                            d_z[gpu],
                            d_value_node_buffer[gpu],
                            node_n,
                            0,
                            sizeof(uint64_t) * 8,
                            stream
                        );

                        cub::DeviceSelect::Unique(
                            d_temp_storage[gpu],
                            max_temp_sizes_bytes[gpu],
                            d_value_node_buffer[gpu],
                            d_edge_buffer[gpu],
                            d_z_unique_n[gpu],
                            node_n,
                            stream
                        );

                        cudaMemcpyAsync(&w_unique_n[gpu], d_w_unique_n[gpu], sizeof(node_t), cudaMemcpyDeviceToHost, stream);
                        cudaMemcpyAsync(&z_unique_n[gpu], d_z_unique_n[gpu], sizeof(node_t), cudaMemcpyDeviceToHost, stream);
                        cudaStreamSynchronize(stream);

                        if(w_unique_n[gpu] == z_unique_n[gpu])
                            break;
                    }

                    current_w = w + (node_n * gpu);
                    current_z = z + (node_n * gpu);

                    cudaMemcpyAsync(current_w, d_w[gpu], node_n * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
                    cudaMemcpyAsync(current_z, d_z[gpu], node_n * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);

                    if(is_equivalent(current_w, current_z, z_c, current_batch_mask, node_n)) {
                        for(i=0; i<node_n; ++i) current_w[i] = node_n;
                        for(i=0; i<node_n; ++i) {
                            if(current_batch_mask[i]) {
                                if(current_w[z_c[i]] == node_n) {
                                    #pragma omp critical 
                                    {
                                        current_w[z_c[i]] = new_node_n++;
                                    }
                                }
                                z_c[i] = current_w[z_c[i]];
                            }
                        }
                        break;
                    }
                }
            }
            cudaStreamDestroy(stream);
            cudaDeviceSynchronize();
        }

        for(node_t i=0; i<node_n; ++i) {
            if(z_c[i] == node_n) { 
                z_c[i] = new_node_n++;
            }
            //z and w are reused to memorize already considered node ids for compaction and weight indexes respectively 
            z[i] = 0; 
            w[i] = edge_n;
        }

        for(node_t i=0; i<start_node_n; ++i) final_z[i] = z_c[final_z[i]];

        new_edge_n = 0;
        node_t current_node;
        uint64_t first_edge_i;
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
    } while(batches > 1 && batches > ceil(new_edge_n / (float)max_batch_edge_n) && node_n > new_node_n);
        
    auto en = std::chrono::steady_clock::now();
    double time_s = std::chrono::duration_cast<std::chrono::microseconds>(en - st).count() / 1000000.0;
    printf("%f\n", time_s);
    printf("%lu\n", batches == 1);
    printf("%lu\n", new_node_n);
    printf("%lu\n", new_edge_n);

    /*for(uint64_t i = 0; i< new_edge_n; ++i) {
        printf("%lu %lu %lu\n", edge_start[i], edge_weight[i], edge_end[i]);
    }
    printf("[%u", final_z[0]);
    for(node_t i = 1; i < start_node_n; ++i) {
        printf(",%u", final_z[i]);
    }
    printf("]\n");*/
    return 0;
}

int is_equivalent(uint64_t* w, uint64_t* z, node_t* z_c, uint8_t* batch_mask, node_t node_n) {
    flat_hash_map<uint64_t, node_t> w_unordered_map;
    node_t w_seen = 1;
    flat_hash_map<uint64_t, node_t> z_unordered_map;
    node_t z_seen = 1;
    w_unordered_map.reserve(node_n);
    z_unordered_map.reserve(node_n);
    node_t z_current_c;
    node_t new_node_n = 0;

    for(node_t i=0; i<node_n; ++i) {
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
                z_current_c = ++new_node_n;
                z_unordered_map[z_val] = z_current_c;
            }
            ++z_seen;
        }
        if(batch_mask[i]) {
            z_c[i] = z_current_c - 1;
        }

        if(w_seen != z_seen) {
            return 0;
        }
    }
    return 1;
}

int read_graph(node_t** edge_start, node_t** edge_end, weight_t** edge_weight, uint64_t* edge_n, node_t* node_n) {
    *node_n = read_uint64();
    *edge_n = read_uint64();
    CHECK_ALLOC( *edge_start = (node_t*)malloc(*edge_n * sizeof(node_t)) );
    CHECK_ALLOC( *edge_end = (node_t*)malloc(*edge_n * sizeof(node_t)) );
    CHECK_ALLOC( *edge_weight = (weight_t*)malloc(*edge_n * sizeof(weight_t)) );
    weight_t tot_weight = 0;
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
