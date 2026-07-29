// ---------------------------------------------------------------------------------------------
//  spark_probe - one shot characterisation of a CPU/GPU machine for the hybrid versions.
//
//  It answers the three questions the hybrid design depends on:
//
//    1. What kind of memory should the shared graph live in? Managed (cudaMallocManaged), pinned
//       host (cudaMallocHost) or device memory? Measured as kernel bandwidth, CPU bandwidth, and -
//       the one that matters - the cost and the *stability* of handing a buffer back and forth,
//       which is exactly what a hybrid level transition does.
//
//    2. What does one level transition cost when there is no work in it? Empty kernel launch,
//       stream synchronise with nothing pending, OpenMP fork/join, OpenMP barrier. The hybrid pays
//       these ~3700 times on 150_10, so microseconds here are tens of milliseconds there.
//
//    3. Which unified memory features does the device actually have?
//
//  Build:  make -C tools spark_probe        Run:  tools/bin/spark_probe
//  Output: a readable table plus PROBE_CSV lines that can be pasted back / collected.
// ---------------------------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <omp.h>

#define CHECK(call)                                                                     \
    do {                                                                                \
        cudaError_t err = (call);                                                        \
        if (err != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err),          \
                    __FILE__, __LINE__);                                                  \
            exit(1);                                                                      \
        }                                                                                \
    } while (0)

static const size_t BUF_MB = 256;
static const size_t N_ELEMS = BUF_MB * 1024 * 1024 / sizeof(int);
static const int REPS = 10;

// ------------------------------------------------------------------ kernels

__global__ void write_kernel(int* p, size_t n, int v) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = v + (int)i;
}

__global__ void read_kernel(const int* p, size_t n, int* sink) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    int acc = 0;
    for (; i < n; i += stride) acc += p[i];
    if (acc == 0x7fffffff) *sink = acc;   // never true, keeps the loads alive
}

__global__ void empty_kernel() {}

// ------------------------------------------------------------------ helpers

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void stats(const double* v, int n, double* mean, double* sd, double* mn, double* mx) {
    double s = 0.0;
    *mn = v[0]; *mx = v[0];
    for (int i = 0; i < n; i++) {
        s += v[i];
        if (v[i] < *mn) *mn = v[i];
        if (v[i] > *mx) *mx = v[i];
    }
    *mean = s / n;
    double q = 0.0;
    for (int i = 0; i < n; i++) q += (v[i] - *mean) * (v[i] - *mean);
    *sd = sqrt(q / n);
}

static double cpu_write(int* p, size_t n) {
    double t0 = now_ms();
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)n; i++) p[i] = (int)i;
    return now_ms() - t0;
}

static long long cpu_read(int* p, size_t n, double* ms) {
    long long acc = 0;
    double t0 = now_ms();
    #pragma omp parallel for schedule(static) reduction(+:acc)
    for (long long i = 0; i < (long long)n; i++) acc += p[i];
    *ms = now_ms() - t0;
    return acc;
}

// ------------------------------------------------------------------ memory kinds

enum MemKind { MEM_DEVICE, MEM_MANAGED, MEM_PINNED, MEM_PAGEABLE, MEM_COUNT };
static const char* mem_name[MEM_COUNT] = { "device", "managed", "pinned host", "pageable host" };

static int* alloc_kind(MemKind k, size_t bytes) {
    int* p = NULL;
    switch (k) {
        case MEM_DEVICE:   CHECK(cudaMalloc((void**)&p, bytes)); break;
        case MEM_MANAGED:  CHECK(cudaMallocManaged((void**)&p, bytes, cudaMemAttachGlobal)); break;
        case MEM_PINNED:   CHECK(cudaMallocHost((void**)&p, bytes)); break;
        case MEM_PAGEABLE: p = (int*)malloc(bytes); break;
        default: break;
    }
    return p;
}

static void free_kind(MemKind k, int* p) {
    switch (k) {
        case MEM_DEVICE:   cudaFree(p); break;
        case MEM_MANAGED:  cudaFree(p); break;
        case MEM_PINNED:   cudaFreeHost(p); break;
        case MEM_PAGEABLE: free(p); break;
        default: break;
    }
}

int main(void) {
    int dev = 0;
    cudaDeviceProp prop;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("=====================================================================\n");
    printf(" device : %s  (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf(" memory : %.1f GB, bus %d bit\n", prop.totalGlobalMem / 1073741824.0, prop.memoryBusWidth);
    printf(" host   : %d OpenMP procs, omp_get_max_threads = %d\n",
           omp_get_num_procs(), omp_get_max_threads());
    printf("---------------------------------------------------------------------\n");
    printf(" unifiedAddressing              %d\n", prop.unifiedAddressing);
    printf(" managedMemory                  %d\n", prop.managedMemory);
    printf(" concurrentManagedAccess        %d\n", prop.concurrentManagedAccess);
    printf(" directManagedMemAccessFromHost %d\n", prop.directManagedMemAccessFromHost);
    printf(" pageableMemoryAccess           %d\n", prop.pageableMemoryAccess);
    printf(" pageableMemoryAccessUsesHostPageTables %d\n",
           prop.pageableMemoryAccessUsesHostPageTables);
    printf(" hostNativeAtomicSupported      %d\n", prop.hostNativeAtomicSupported);
    printf(" canMapHostMemory               %d\n", prop.canMapHostMemory);
    printf("=====================================================================\n\n");

    printf("PROBE_CSV,device,%s,%d,%d,%d,%d,%d,%d\n", prop.name,
           prop.unifiedAddressing, prop.managedMemory, prop.concurrentManagedAccess,
           prop.directManagedMemAccessFromHost, prop.pageableMemoryAccess,
           prop.pageableMemoryAccessUsesHostPageTables);

    const size_t bytes = N_ELEMS * sizeof(int);
    int* sink = NULL;
    CHECK(cudaMalloc((void**)&sink, sizeof(int)));

    int blocks = prop.multiProcessorCount * 16;

    // ---------------------------------------------------------------- bandwidths
    printf("\n--- bandwidth over %zu MB (%d reps, mean +- sd) ---\n", BUF_MB, REPS);
    printf("%-14s %14s %14s %14s %14s\n", "memory", "GPU write", "GPU read", "CPU write", "CPU read");

    for (int k = 0; k < MEM_COUNT; k++) {
        MemKind kind = (MemKind)k;

        // a kernel cannot touch pageable host memory unless the device supports it
        bool gpu_ok = !(kind == MEM_PAGEABLE && !prop.pageableMemoryAccess);
        bool cpu_ok = (kind != MEM_DEVICE);

        int* p = alloc_kind(kind, bytes);
        if (!p) { printf("%-14s allocation failed\n", mem_name[k]); continue; }

        double gw[REPS], gr[REPS], cw[REPS], cr[REPS];
        for (int r = 0; r < REPS; r++) { gw[r] = gr[r] = cw[r] = cr[r] = 0.0; }

        for (int r = 0; r < REPS; r++) {
            if (gpu_ok) {
                CHECK(cudaDeviceSynchronize());
                double t0 = now_ms();
                write_kernel<<<blocks, 256>>>(p, N_ELEMS, r);
                CHECK(cudaDeviceSynchronize());
                gw[r] = now_ms() - t0;

                t0 = now_ms();
                read_kernel<<<blocks, 256>>>(p, N_ELEMS, sink);
                CHECK(cudaDeviceSynchronize());
                gr[r] = now_ms() - t0;
            }
            if (cpu_ok) {
                cw[r] = cpu_write(p, N_ELEMS);
                double ms; cpu_read(p, N_ELEMS, &ms); cr[r] = ms;
            }
        }

        double m, sd, mn, mx;
        double gb = bytes / 1073741824.0;
        printf("%-14s", mem_name[k]);
        if (gpu_ok) { stats(gw, REPS, &m, &sd, &mn, &mx); printf(" %8.1f GB/s", gb / (m / 1e3)); }
        else printf(" %13s", "n/a");
        if (gpu_ok) { stats(gr, REPS, &m, &sd, &mn, &mx); printf(" %8.1f GB/s", gb / (m / 1e3)); }
        else printf(" %13s", "n/a");
        if (cpu_ok) { stats(cw, REPS, &m, &sd, &mn, &mx); printf(" %8.1f GB/s", gb / (m / 1e3)); }
        else printf(" %13s", "n/a");
        if (cpu_ok) { stats(cr, REPS, &m, &sd, &mn, &mx); printf(" %8.1f GB/s", gb / (m / 1e3)); }
        else printf(" %13s", "n/a");
        printf("\n");

        if (gpu_ok) { stats(gw, REPS, &m, &sd, &mn, &mx);
            printf("PROBE_CSV,bw_gpu_write,%s,%.3f,%.3f,%.3f,%.3f\n", mem_name[k], m, sd, mn, mx); }
        if (gpu_ok) { stats(gr, REPS, &m, &sd, &mn, &mx);
            printf("PROBE_CSV,bw_gpu_read,%s,%.3f,%.3f,%.3f,%.3f\n", mem_name[k], m, sd, mn, mx); }
        if (cpu_ok) { stats(cw, REPS, &m, &sd, &mn, &mx);
            printf("PROBE_CSV,bw_cpu_write,%s,%.3f,%.3f,%.3f,%.3f\n", mem_name[k], m, sd, mn, mx); }
        if (cpu_ok) { stats(cr, REPS, &m, &sd, &mn, &mx);
            printf("PROBE_CSV,bw_cpu_read,%s,%.3f,%.3f,%.3f,%.3f\n", mem_name[k], m, sd, mn, mx); }

        free_kind(kind, p);
    }

    // ---------------------------------------------------------------- ping pong
    // This is the hybrid's level transition: the GPU writes a slice, the CPU reads it, over and
    // over. On a machine that migrates pages this is where the time (and the variance) appears.
    printf("\n--- hand over: GPU writes 64 MB, CPU reads it, %d rounds (ms per round) ---\n", REPS);
    printf("%-14s %10s %10s %10s %10s\n", "memory", "mean", "sd", "min", "max");

    const size_t PP_ELEMS = 64ull * 1024 * 1024 / sizeof(int);
    for (int k = 1; k < MEM_COUNT; k++) {   // device memory cannot be read by the CPU
        MemKind kind = (MemKind)k;
        if (kind == MEM_PAGEABLE && !prop.pageableMemoryAccess) continue;

        int* p = alloc_kind(kind, PP_ELEMS * sizeof(int));
        if (!p) continue;

        double t[REPS];
        for (int r = 0; r < REPS; r++) {
            double t0 = now_ms();
            write_kernel<<<blocks, 256>>>(p, PP_ELEMS, r);
            CHECK(cudaDeviceSynchronize());
            double ms;
            cpu_read(p, PP_ELEMS, &ms);
            t[r] = now_ms() - t0;
        }

        double m, sd, mn, mx;
        stats(t, REPS, &m, &sd, &mn, &mx);
        printf("%-14s %10.3f %10.3f %10.3f %10.3f\n", mem_name[k], m, sd, mn, mx);
        printf("PROBE_CSV,pingpong,%s,%.3f,%.3f,%.3f,%.3f\n", mem_name[k], m, sd, mn, mx);

        free_kind(kind, p);
    }

    // ---------------------------------------------------------------- per level overheads
    printf("\n--- per level overheads (us, mean of 2000) ---\n");

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    const int IT = 2000;

    CHECK(cudaDeviceSynchronize());
    double t0 = now_ms();
    for (int i = 0; i < IT; i++) empty_kernel<<<1, 32, 0, stream>>>();
    double issue_us = (now_ms() - t0) * 1e3 / IT;
    CHECK(cudaStreamSynchronize(stream));

    t0 = now_ms();
    for (int i = 0; i < IT; i++) {
        empty_kernel<<<1, 32, 0, stream>>>();
        CHECK(cudaStreamSynchronize(stream));
    }
    double launch_sync_us = (now_ms() - t0) * 1e3 / IT;

    t0 = now_ms();
    for (int i = 0; i < IT; i++) CHECK(cudaStreamSynchronize(stream));
    double idle_sync_us = (now_ms() - t0) * 1e3 / IT;

    t0 = now_ms();
    for (int i = 0; i < IT; i++) CHECK(cudaDeviceSynchronize());
    double idle_devsync_us = (now_ms() - t0) * 1e3 / IT;

    // OpenMP: fork/join per level vs a barrier inside a persistent region
    volatile int guard = 0;
    for (int nt = 2; nt <= omp_get_num_procs(); nt *= 2) {
        t0 = now_ms();
        for (int i = 0; i < IT; i++) {
            #pragma omp parallel for schedule(static) num_threads(nt)
            for (int t = 0; t < nt; t++) guard += t;
        }
        double fork_us = (now_ms() - t0) * 1e3 / IT;

        t0 = now_ms();
        #pragma omp parallel num_threads(nt)
        {
            for (int i = 0; i < IT; i++) {
                #pragma omp for schedule(static)
                for (int t = 0; t < nt; t++) guard += t;
            }
        }
        double barrier_us = (now_ms() - t0) * 1e3 / IT;

        printf(" omp fork/join (%2d threads)        %8.2f us    omp barrier in region %8.2f us\n",
               nt, fork_us, barrier_us);
        printf("PROBE_CSV,omp,%d,%.3f,%.3f\n", nt, fork_us, barrier_us);
    }

    printf(" empty kernel, issue only          %8.2f us\n", issue_us);
    printf(" empty kernel + stream sync        %8.2f us\n", launch_sync_us);
    printf(" stream sync, nothing pending      %8.2f us\n", idle_sync_us);
    printf(" device sync, nothing pending      %8.2f us\n", idle_devsync_us);
    printf("PROBE_CSV,launch,issue,%.3f\n", issue_us);
    printf("PROBE_CSV,launch,launch_sync,%.3f\n", launch_sync_us);
    printf("PROBE_CSV,launch,idle_stream_sync,%.3f\n", idle_sync_us);
    printf("PROBE_CSV,launch,idle_device_sync,%.3f\n", idle_devsync_us);

    printf("\ndone (guard %d)\n", guard);

    cudaStreamDestroy(stream);
    cudaFree(sink);
    return 0;
}
