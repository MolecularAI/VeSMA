// Exact nearest-stock Tanimoto distance on Apple Metal.
//
// The device twin of `vesma.packed_circularfp.nearest_stock_distance`, which is exact against the
// whole catalogue and therefore costs `queries * stock * n_bits / 64` popcounts however it is
// written -- ten CPU-minutes for one AO* pass against a million-row stock. Nothing about the answer
// changes here; only where the popcounts happen.
//
// Two things this does that the CUDA kernel in `cuda_circularfp.py` does not, both because a
// million-row catalogue makes them matter:
//
//   * A threadgroup holds a *tile* of queries and streams the stock past them, rather than giving
//     each query its own threadgroup. One-threadgroup-per-query reads the entire catalogue once per
//     query: at 19k queries against 253 MB that is 4.9 TB of traffic. A 64-query tile reads it 302
//     times instead of 19,296, which is the difference between bandwidth-bound and compute-bound.
//
//   * The running nearest is compared by cross-multiplication rather than by dividing. `i/u` beats
//     `bi/bu` exactly when `i*bu > bi*u`, so the inner loop carries no divide and the comparison is
//     exact integer arithmetic -- one divide per query at the end instead of one per (query, row)
//     pair, which at 1.9e10 pairs was going to be the dominant term.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static char g_last_error[1024] = {0};

static const char* kShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct NearestParams {
    uint query_count;
    uint stock_count;
    uint quads_per_row;
    uint query_tile;
    uint stock_tile_rows;
};

struct FingerprintParams {
    uint molecule_count;
    uint radius;
    uint n_bits;
};

inline ulong circular_mix(ulong value) {
    value += 0x9E3779B97F4A7C15ul;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ul;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBul;
    return value ^ (value >> 31);
}

inline ulong circular_combine(ulong left, ulong right) {
    return circular_mix(left ^ (right + 0x517CC1B727220A95ul + (left << 6) + (left >> 2)));
}

inline uint circular_first_set(ulong bits) {
    const uint low = (uint)bits;
    return low ? ctz(low) : 32u + ctz((uint)(bits >> 32));
}

inline void circular_set_bit(device atomic_uint* words, uint n_bits, ulong label) {
    const ulong bit = label % (ulong)n_bits;
    atomic_fetch_or_explicit(words + (bit >> 5), 1u << (bit & 31u), memory_order_relaxed);
}

// One threadgroup per packed molecule, one lane per atom. The host AO* batch already uses this
// layout, so this kernel needs no graph-object conversion or RDKit call.
kernel void circular_fingerprint_kernel(
    device const uint* atoms_global [[buffer(0)]],
    device const uchar* bonds_global [[buffer(1)]],
    device const uint* counts [[buffer(2)]],
    device atomic_uint* out [[buffer(3)]],
    constant FingerprintParams& params [[buffer(4)]],
    uint molecule [[threadgroup_position_in_grid]],
    uint atom [[thread_index_in_threadgroup]]) {

    if (molecule >= params.molecule_count) return;
    const uint atom_count = counts[molecule];
    device const uint* atoms = atoms_global + (size_t)molecule * 64u * 14u;
    device const uchar* bonds = bonds_global + (size_t)molecule * 64u * 64u;
    device atomic_uint* words = out + (size_t)molecule * (params.n_bits / 32u);
    threadgroup ulong previous[64];
    threadgroup ulong next[64];
    threadgroup ulong rows[64];

    if (atom < atom_count) {
        ulong row = 0ul;
        for (uint neighbor = 0; neighbor < atom_count; ++neighbor) {
            if (bonds[atom * 64u + neighbor] != 0u) row |= 1ul << neighbor;
        }
        rows[atom] = row;
        device const uint* packed = atoms + atom * 14u;
        ulong label = 0xA70F4A7D9E370003ul;
        label = circular_combine(label, (ulong)packed[0] | ((ulong)packed[1] << 32));
        label = circular_combine(label, (ulong)packed[2] | ((ulong)packed[3] << 32));
        label = circular_combine(label, (ulong)packed[4]);
        label = circular_combine(label, (ulong)packed[5]);
        label = circular_combine(label, (ulong)packed[7]);
        label = circular_combine(label, (ulong)(popcount((uint)row) + popcount((uint)(row >> 32))));
        previous[atom] = label;
        circular_set_bit(words, params.n_bits, label);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint round = 0; round < params.radius; ++round) {
        if (atom < atom_count) {
            ulong aggregate = 0ul;
            ulong neighbors = rows[atom];
            while (neighbors != 0ul) {
                const uint neighbor = circular_first_set(neighbors);
                neighbors &= neighbors - 1ul;
                const ulong bond = (ulong)(bonds[atom * 64u + neighbor] & 0x0fu);
                aggregate += circular_combine(previous[neighbor], bond);
            }
            const ulong degree = (ulong)(popcount((uint)rows[atom]) + popcount((uint)(rows[atom] >> 32)));
            next[atom] = circular_mix(circular_combine(circular_combine(previous[atom], degree), aggregate));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atom < atom_count) {
            previous[atom] = next[atom];
            circular_set_bit(words, params.n_bits, previous[atom]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Keep the better of two candidate ratios without dividing: `i/u` beats `bi/bu` exactly when
// `i*bu > bi*u`, which is exact in integers where a float comparison of two quotients is not.
// 64-bit products because 32 bits only just holds `n_bits * n_bits` and the width is a caller's
// choice; the multiply is once per candidate row, not once per word.
inline void keep_if_nearer(
    uint intersection,
    uint query_bits,
    uint stock_bits,
    thread uint& best_intersection,
    thread uint& best_union) {
    // |q | s| = |q| + |s| - |q & s|, so the union costs an add rather than a second popcount over
    // the pair -- the same identity the host path uses, and worth the same third of the work here.
    const uint union_bits = query_bits + stock_bits - intersection;
    if (union_bits == 0u) return;
    if ((ulong)intersection * (ulong)best_union >
        (ulong)best_intersection * (ulong)union_bits) {
        best_intersection = intersection;
        best_union = union_bits;
    }
}

inline uint fold_quad(uint4 accumulated) {
    return accumulated.x + accumulated.y + accumulated.z + accumulated.w;
}

// Fingerprints arrive as uint64 words and are read here as `uint4`. Two reasons, both measured: an
// Apple GPU is a 32-bit machine, so `popcount` on a uint is one instruction where a ulong popcount
// is two plus a fold; and the trip count is a runtime parameter, so the loop cannot be unrolled and
// its overhead is paid per iteration -- four words per iteration divides that overhead by four and
// turns two scalar loads into two 16-byte vector loads. Tanimoto does not care how the bits are
// grouped, so the cheapest grouping is free to take.
kernel void nearest_stock_similarity_kernel(
    device const uint4* queries [[buffer(0)]],
    device const uint* query_popcounts [[buffer(1)]],
    device const uint4* stock [[buffer(2)]],
    device const uint* stock_popcounts [[buffer(3)]],
    device atomic_uint* best_similarity [[buffer(4)]],
    constant NearestParams& params [[buffer(5)]],
    threadgroup uint4* shared_queries [[threadgroup(0)]],
    uint3 group3 [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group_size3 [[threads_per_threadgroup]]) {

    const uint tid = tid3.x;
    const uint group_size = group_size3.x;
    const uint quads = params.quads_per_row;
    const uint query_base = group3.x * params.query_tile;

    // Transposed to [quad][slot] so that the inner loop's `shared_queries[quad * tile + tid]` is
    // consecutive across the simdgroup and therefore bank-conflict free -- the natural [slot][quad]
    // layout would put every lane of a simdgroup in the same bank. Slots past the end of the batch
    // are zeroed rather than skipped, so the inner loop needs no bounds test.
    for (uint idx = tid; idx < params.query_tile * quads; idx += group_size) {
        const uint slot = idx / quads;
        const uint quad = idx % quads;
        const uint row = query_base + slot;
        shared_queries[quad * params.query_tile + slot] =
            (row < params.query_count) ? queries[(size_t)row * quads + quad] : uint4(0u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint query = query_base + tid;
    if (query >= params.query_count) return;

    const uint query_bits = query_popcounts[query];
    // The ratio 0/1, which is what an all-zero fingerprint scores and therefore the honest floor:
    // no candidate with an empty intersection can improve on it.
    uint best_intersection = 0u;
    uint best_union = 1u;

    const uint start = group3.y * params.stock_tile_rows;
    uint stop = start + params.stock_tile_rows;
    if (stop > params.stock_count) stop = params.stock_count;

    // Two candidate rows per iteration, against one read of the query. The query word is the same
    // for both, so pairing the rows halves the threadgroup-memory traffic per unit of work and gives
    // the two dependency chains something to interleave; measured at 1.3x over one row at a time,
    // which is the largest single win left after vectorising. Accumulators are folded once per row
    // rather than summed per quad, another three adds per quad saved.
    uint row = start;
    for (; row + 1u < stop; row += 2u) {
        device const uint4* first = stock + (size_t)row * quads;
        device const uint4* second = first + quads;
        uint4 accumulated_first = uint4(0u);
        uint4 accumulated_second = uint4(0u);
#pragma clang loop unroll_count(4)
        for (uint quad = 0; quad < quads; ++quad) {
            const uint4 mine = shared_queries[quad * params.query_tile + tid];
            accumulated_first += popcount(first[quad] & mine);
            accumulated_second += popcount(second[quad] & mine);
        }
        keep_if_nearer(fold_quad(accumulated_first), query_bits, stock_popcounts[row],
                       best_intersection, best_union);
        keep_if_nearer(fold_quad(accumulated_second), query_bits, stock_popcounts[row + 1u],
                       best_intersection, best_union);
    }
    // A catalogue of odd length, or a tile that ends on one.
    if (row < stop) {
        device const uint4* candidate = stock + (size_t)row * quads;
        uint4 accumulated = uint4(0u);
#pragma clang loop unroll_count(4)
        for (uint quad = 0; quad < quads; ++quad) {
            accumulated += popcount(
                candidate[quad] & shared_queries[quad * params.query_tile + tid]
            );
        }
        keep_if_nearer(fold_quad(accumulated), query_bits, stock_popcounts[row],
                       best_intersection, best_union);
    }

    // Similarity is in [0, 1], so its IEEE-754 bit pattern orders exactly as the float does and an
    // integer atomic max is a float max. Order-independent, hence deterministic however the stock
    // tiles happen to retire -- which is why this needs no partial buffer and no second pass.
    const float similarity = (float)best_intersection / (float)best_union;
    atomic_fetch_max_explicit(
        best_similarity + query, as_type<uint>(similarity), memory_order_relaxed
    );
}
)";

static id<MTLDevice> g_device = nil;
static id<MTLComputePipelineState> g_nearest_pipeline = nil;
static id<MTLComputePipelineState> g_fingerprint_pipeline = nil;
static id<MTLCommandQueue> g_queue = nil;
static dispatch_once_t g_once;
static bool g_init_failed = false;

// The stock is uploaded once and reused across every call carrying the same token. A 988k-molecule
// catalogue is 253 MB, an AO* search scores against it seventeen times per target, and re-copying
// it each time would cost more than the scan does. The token is the caller's identity for a
// catalogue rather than anything derived from the pointer: a freed array can be replaced by another
// at the same address with the same shape, and that would be a silently wrong answer.
static id<MTLBuffer> g_stock = nil;
static id<MTLBuffer> g_stock_popcounts = nil;
static uint64_t g_stock_token = 0;
static uint32_t g_stock_count = 0;
static uint32_t g_stock_words = 0;

static void ensure_metal(void) {
    dispatch_once(&g_once, ^{
        @autoreleasepool {
            g_device = MTLCreateSystemDefaultDevice();
            if (!g_device) {
                snprintf(g_last_error, sizeof(g_last_error), "No Metal device found");
                g_init_failed = true; return;
            }
            NSError* error = nil;
            // Fast math off for one instruction's sake: it is on by default and turns the single
            // `i / u` into a reciprocal multiply, which is a 1-ulp error in the only float the
            // kernel computes. That ulp is the whole difference between "agrees with the host
            // reference to within tolerance" and "is bit-identical to it", and the kernel is
            // otherwise integer work that fast math cannot help.
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            if (@available(macOS 15.0, iOS 18.0, *)) {
                options.mathMode = MTLMathModeSafe;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
            id<MTLLibrary> library = [g_device
                newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource]
                             options:options
                               error:&error];
            if (!library) {
                snprintf(g_last_error, sizeof(g_last_error), "Shader compilation failed: %s",
                         [[error localizedDescription] UTF8String]);
                g_init_failed = true; return;
            }
            id<MTLFunction> function =
                [library newFunctionWithName:@"nearest_stock_similarity_kernel"];
            if (!function) {
                snprintf(g_last_error, sizeof(g_last_error),
                         "Function nearest_stock_similarity_kernel not found");
                g_init_failed = true; return;
            }
            g_nearest_pipeline = [g_device newComputePipelineStateWithFunction:function
                                                                        error:&error];
            if (!g_nearest_pipeline) {
                snprintf(g_last_error, sizeof(g_last_error), "Pipeline creation failed: %s",
                         [[error localizedDescription] UTF8String]);
                g_init_failed = true; return;
            }
            id<MTLFunction> fingerprint_function =
                [library newFunctionWithName:@"circular_fingerprint_kernel"];
            g_fingerprint_pipeline = [g_device
                newComputePipelineStateWithFunction:fingerprint_function error:&error];
            if (!g_fingerprint_pipeline) {
                snprintf(g_last_error, sizeof(g_last_error), "Fingerprint pipeline creation failed: %s",
                         [[error localizedDescription] UTF8String]);
                g_init_failed = true; return;
            }
            g_queue = [g_device newCommandQueue];
        }
    });
}

extern "C" int vesma_metal_circular_fingerprints(
    const uint32_t* atoms,
    const uint8_t* bonds,
    const uint32_t* atom_counts,
    uint32_t molecule_count,
    uint32_t radius,
    uint32_t n_bits,
    uint64_t* out,
    double* out_kernel_seconds) {

    if (out_kernel_seconds) *out_kernel_seconds = 0.0;
    ensure_metal();
    if (g_init_failed) return 0;
    if (molecule_count == 0u) return 1;
    if (n_bits == 0u || (n_bits & 63u) != 0u) {
        snprintf(g_last_error, sizeof(g_last_error), "n_bits must be a positive multiple of 64");
        return 0;
    }
    @autoreleasepool {
        const NSUInteger atoms_bytes = (NSUInteger)molecule_count * 64u * 14u * sizeof(uint32_t);
        const NSUInteger bonds_bytes = (NSUInteger)molecule_count * 64u * 64u;
        const NSUInteger counts_bytes = (NSUInteger)molecule_count * sizeof(uint32_t);
        const NSUInteger out_bytes = (NSUInteger)molecule_count * (n_bits / 8u);
        id<MTLBuffer> buf_atoms = [g_device newBufferWithBytes:atoms length:atoms_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_bonds = [g_device newBufferWithBytes:bonds length:bonds_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_counts = [g_device newBufferWithBytes:atom_counts length:counts_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_out = [g_device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        struct {
            uint32_t molecule_count;
            uint32_t radius;
            uint32_t n_bits;
        } params = {molecule_count, radius, n_bits};
        id<MTLBuffer> buf_params = [g_device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
        if (!buf_atoms || !buf_bonds || !buf_counts || !buf_out || !buf_params) {
            snprintf(g_last_error, sizeof(g_last_error), "Could not allocate fingerprint buffers for %u molecules", molecule_count);
            return 0;
        }
        memset([buf_out contents], 0, out_bytes);
        id<MTLCommandBuffer> commandBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        commandBuffer.label = @"vesma.circular_fingerprint";
        encoder.label = @"circular_fingerprint";
        [encoder setComputePipelineState:g_fingerprint_pipeline];
        [encoder setBuffer:buf_atoms offset:0 atIndex:0];
        [encoder setBuffer:buf_bonds offset:0 atIndex:1];
        [encoder setBuffer:buf_counts offset:0 atIndex:2];
        [encoder setBuffer:buf_out offset:0 atIndex:3];
        [encoder setBuffer:buf_params offset:0 atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(molecule_count, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        // The device's own clock, as `metal_retro.mm` reads it. This translation unit had no GPU
        // timestamps at all, so neither of its kernels could be told apart from the host work
        // around it -- and the nearest-stock scan turned out to be the largest single cost in a
        // search once the rest was named.
        if (out_kernel_seconds) {
            *out_kernel_seconds = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
        }
        if ([commandBuffer status] == MTLCommandBufferStatusError) {
            snprintf(g_last_error, sizeof(g_last_error), "Fingerprint kernel failed: %s",
                     [[[commandBuffer error] localizedDescription] UTF8String]);
            return 0;
        }
        memcpy(out, [buf_out contents], out_bytes);
        return 1;
    }
}

extern "C" uint32_t vesma_metal_circularfp_max_threadgroup_memory(void) {
    ensure_metal();
    if (g_init_failed) return 0;
    return (uint32_t)[g_device maxThreadgroupMemoryLength];
}

extern "C" uint32_t vesma_metal_circularfp_max_threads_per_group(void) {
    ensure_metal();
    if (g_init_failed) return 0;
    return (uint32_t)[g_nearest_pipeline maxTotalThreadsPerThreadgroup];
}

// Writes `query_count` floats: the largest Tanimoto similarity between each query and any stock row.
// The `weight * (1 - s)` that turns a similarity into a heuristic estimate is left to the caller, so
// this function computes exactly one thing and the host and device paths apply that arithmetic in
// the same place -- which is what makes the parity test a test of the kernel.
extern "C" int vesma_metal_nearest_stock_similarity(
    const uint64_t* stock,
    const uint32_t* stock_popcounts,
    uint32_t stock_count,
    const uint64_t* queries,
    const uint32_t* query_popcounts,
    uint32_t query_count,
    uint32_t word_count,
    uint64_t stock_token,
    uint32_t query_tile,
    uint32_t stock_tile_rows,
    float* out,
    double* out_kernel_seconds) {

    if (out_kernel_seconds) *out_kernel_seconds = 0.0;
    ensure_metal();
    if (g_init_failed) return 0;

    if (word_count == 0u || (word_count & 1u) != 0u) {
        // The inner loop reads four 32-bit words at a time, so a row is a whole number of `uint4`.
        // The caller pads rather than this branching, because padding with zero words changes
        // neither the intersection nor the union and no production width is odd anyway.
        snprintf(g_last_error, sizeof(g_last_error),
                 "word_count must be a positive even number of uint64 words, got %u", word_count);
        return 0;
    }

    // Held across the dispatch; the lock-width note in `metal_retro.mm`'s construct entry says
    // when that is required and when the matcher's narrow form is enough.
    // The resident stock and its token are process-global. ctypes releases the GIL while calling
    // this function, so another Python thread can otherwise replace the buffers after the token
    // check but before the encoder binds them. Keep the cache decision and command completion
    // together; command encoding retains the chosen buffers, but only once they are bound.
    @synchronized (g_device) {
    @autoreleasepool {
        const uint32_t quads = word_count / 2u;
        const NSUInteger stock_bytes =
            (NSUInteger)stock_count * word_count * sizeof(uint64_t);
        const NSUInteger stock_pop_bytes = (NSUInteger)stock_count * sizeof(uint32_t);

        if (stock_token == 0 || stock_token != g_stock_token) {
            g_stock_token = 0;
            g_stock = [g_device newBufferWithBytes:stock
                                           length:stock_bytes
                                          options:MTLResourceStorageModeShared];
            g_stock_popcounts = [g_device newBufferWithBytes:stock_popcounts
                                                     length:stock_pop_bytes
                                                    options:MTLResourceStorageModeShared];
            if (!g_stock || !g_stock_popcounts) {
                snprintf(g_last_error, sizeof(g_last_error),
                         "Could not allocate %llu bytes for a stock of %u rows",
                         (unsigned long long)stock_bytes, stock_count);
                return 0;
            }
            g_stock_token = stock_token;
            g_stock_count = stock_count;
            g_stock_words = word_count;
        } else if (g_stock_count != stock_count || g_stock_words != word_count) {
            snprintf(g_last_error, sizeof(g_last_error),
                     "stock token %llu was uploaded as %u rows of %u words and is now offered as "
                     "%u rows of %u words",
                     (unsigned long long)stock_token, g_stock_count, g_stock_words,
                     stock_count, word_count);
            return 0;
        }

        const NSUInteger query_bytes =
            (NSUInteger)query_count * word_count * sizeof(uint64_t);
        id<MTLBuffer> buf_queries = [g_device newBufferWithBytes:queries
                                                         length:query_bytes
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_query_popcounts =
            [g_device newBufferWithBytes:query_popcounts
                                 length:(NSUInteger)query_count * sizeof(uint32_t)
                                options:MTLResourceStorageModeShared];
        const NSUInteger out_bytes = (NSUInteger)query_count * sizeof(float);
        id<MTLBuffer> buf_out = [g_device newBufferWithLength:out_bytes
                                                     options:MTLResourceStorageModeShared];
        if (!buf_queries || !buf_query_popcounts || !buf_out) {
            snprintf(g_last_error, sizeof(g_last_error),
                     "Could not allocate device buffers for %u queries", query_count);
            return 0;
        }
        // Zero is the bit pattern of 0.0f, which is the identity for the atomic max below.
        memset([buf_out contents], 0, out_bytes);

        const NSUInteger shared_bytes = (NSUInteger)query_tile * quads * sizeof(uint32_t) * 4u;
        if (shared_bytes > [g_device maxThreadgroupMemoryLength]) {
            snprintf(g_last_error, sizeof(g_last_error),
                     "a %u-query tile of %u-word fingerprints needs %llu bytes of threadgroup "
                     "memory and the device offers %llu",
                     query_tile, word_count, (unsigned long long)shared_bytes,
                     (unsigned long long)[g_device maxThreadgroupMemoryLength]);
            return 0;
        }
        if (query_tile > [g_nearest_pipeline maxTotalThreadsPerThreadgroup]) {
            snprintf(g_last_error, sizeof(g_last_error),
                     "a query tile of %u exceeds the pipeline's %llu threads per threadgroup",
                     query_tile,
                     (unsigned long long)[g_nearest_pipeline maxTotalThreadsPerThreadgroup]);
            return 0;
        }

        struct {
            uint32_t query_count;
            uint32_t stock_count;
            uint32_t quads_per_row;
            uint32_t query_tile;
            uint32_t stock_tile_rows;
        } params = {query_count, stock_count, quads, query_tile, stock_tile_rows};
        id<MTLBuffer> buf_params = [g_device newBufferWithBytes:&params
                                                        length:sizeof(params)
                                                       options:MTLResourceStorageModeShared];

        const uint32_t query_groups = (query_count + query_tile - 1u) / query_tile;
        const uint32_t stock_groups = (stock_count + stock_tile_rows - 1u) / stock_tile_rows;

        id<MTLCommandBuffer> commandBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        commandBuffer.label = @"vesma.nearest_stock_similarity";
        encoder.label = @"nearest_stock_similarity";
        [encoder setComputePipelineState:g_nearest_pipeline];
        [encoder setBuffer:buf_queries offset:0 atIndex:0];
        [encoder setBuffer:buf_query_popcounts offset:0 atIndex:1];
        [encoder setBuffer:g_stock offset:0 atIndex:2];
        [encoder setBuffer:g_stock_popcounts offset:0 atIndex:3];
        [encoder setBuffer:buf_out offset:0 atIndex:4];
        [encoder setBuffer:buf_params offset:0 atIndex:5];
        [encoder setThreadgroupMemoryLength:shared_bytes atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(query_groups, stock_groups, 1)
               threadsPerThreadgroup:MTLSizeMake(query_tile, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (out_kernel_seconds) {
            *out_kernel_seconds = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
        }

        if ([commandBuffer status] == MTLCommandBufferStatusError) {
            snprintf(g_last_error, sizeof(g_last_error), "Command buffer error: %s",
                     [[[commandBuffer error] localizedDescription] UTF8String]);
            return 0;
        }

        memcpy(out, [buf_out contents], out_bytes);
        return 1;
    }
    }
}

extern "C" const char* vesma_metal_circularfp_last_error(void) {
    return g_last_error;
}
