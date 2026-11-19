#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static char g_last_error[1024] = {0};

static const char* kShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint BUCKET_SIZE = 64u;
constant constexpr uint ATOM_WORD_COUNT = 14u;
constant constexpr uchar BOND_NONE = 0;
constant constexpr uchar BOND_CLASS_MASK = 0x0f;
constant constexpr ulong SINGLE_BOND_CLASS = 1UL;
constant constexpr ulong DOUBLE_BOND_CLASS = 2UL;
constant constexpr ulong TRIPLE_BOND_CLASS = 3UL;
constant constexpr ulong FLEX_BOND_CLASS = 4UL;
constant constexpr ulong ATOM_FEATURE = 0xA70F4A7D9E370001UL;
constant constexpr ulong PATH_FEATURE = 0xA70F4A7D9E370002UL;
constant constexpr ulong WL_COMBINE_CONSTANT = 0x517CC1B727220A95UL;
constant constexpr ulong SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15UL;
constant constexpr ulong SPLITMIX64_MUL1 = 0xBF58476D1CE4E5B9UL;
constant constexpr ulong SPLITMIX64_MUL2 = 0x94D049BB133111EBUL;
constant constexpr uint FP_WORD_COUNT = 32u;
constant constexpr uint FP_TILE_SIZE = 32u;

inline ulong mix_uint64_linear(ulong value) {
    value += SPLITMIX64_GAMMA;
    value = (value ^ (value >> 30)) * SPLITMIX64_MUL1;
    value = (value ^ (value >> 27)) * SPLITMIX64_MUL2;
    return value ^ (value >> 31);
}

inline ulong combine_uint64_linear(ulong left, ulong right) {
    ulong combined = left ^ (right + WL_COMBINE_CONSTANT + (left << 6) + (left >> 2));
    return mix_uint64_linear(combined);
}

inline int popcount_atom_z_words(const thread uint* atom) {
    return popcount(atom[0]) + popcount(atom[1]) + popcount(atom[2]) + popcount(atom[3]);
}

inline int exact_single_z_from_atom(const thread uint* atom) {
    if (popcount_atom_z_words(atom) != 1) return -1;
    if (atom[0] != 0u) return (int)ctz(atom[0]);
    if (atom[1] != 0u) return (int)ctz(atom[1]) + 32;
    if (atom[2] != 0u) return (int)ctz(atom[2]) + 64;
    return (int)ctz(atom[3]) + 96;
}

inline ulong atom_feature_linear(int z) {
    return combine_uint64_linear(ATOM_FEATURE, (ulong)z);
}

inline ulong path_hash_linear(
    const threadgroup int* labels,
    const thread int* path,
    const thread ulong* edge_classes,
    int length,
    bool reverse
) {
    ulong value = PATH_FEATURE;
    if (reverse) {
        for (int i = length; i >= 0; --i) {
            value = combine_uint64_linear(value, (ulong)labels[path[i]]);
        }
        for (int i = length - 1; i >= 0; --i) {
            value = combine_uint64_linear(value, edge_classes[i]);
        }
    } else {
        for (int i = 0; i <= length; ++i) {
            value = combine_uint64_linear(value, (ulong)labels[path[i]]);
        }
        for (int i = 0; i < length; ++i) {
            value = combine_uint64_linear(value, edge_classes[i]);
        }
    }
    return value;
}

inline void set_fingerprint_bit(device atomic_uint* out, int word_count, ulong feature) {
    ulong bit = feature % (ulong)(word_count * 64);
    atomic_fetch_or_explicit(out + (bit >> 5), 1U << (bit & 31U), memory_order_relaxed);
}

// `query` is accepted and ignored, and the SINGLE_/DOUBLE_/TRIPLE_BOND_CLASS constants above
// are unreached. That is faithful to `invariants.bond_feature_classes`, whose docstring explains
// the choice and why the prefilter stays subset-sound; do not "fix" it on one backend alone.
inline int bond_feature_classes_linear(uchar flags, bool query, thread ulong* classes) {
    if ((flags & BOND_CLASS_MASK) == BOND_NONE) return 0;
    classes[0] = FLEX_BOND_CLASS;
    return 1;
}

inline void emit_path_feature_variants(
    device atomic_uint* out,
    int word_count,
    const threadgroup uchar* bonds,
    const threadgroup int* labels,
    const thread int* path,
    int length,
    bool query
) {
    ulong options[7][4];
    int option_counts[7];
    for (int i = 0; i <= length; ++i) {
        if (labels[path[i]] < 0) return;
    }
    for (int i = 0; i < length; ++i) {
        int left = path[i];
        int right = path[i + 1];
        option_counts[i] = bond_feature_classes_linear(bonds[left * BUCKET_SIZE + right], query, options[i]);
        if (option_counts[i] == 0) return;
    }

    int choice[7];
    for (int i = 0; i < length; ++i) choice[i] = 0;
    while (true) {
        ulong edge_classes[7];
        for (int i = 0; i < length; ++i) {
            edge_classes[i] = options[i][choice[i]];
        }
        ulong forward = path_hash_linear(labels, path, edge_classes, length, false);
        ulong reverse = path_hash_linear(labels, path, edge_classes, length, true);
        set_fingerprint_bit(out, word_count, forward < reverse ? forward : reverse);

        int pos = length - 1;
        while (pos >= 0) {
            choice[pos] += 1;
            if (choice[pos] < option_counts[pos]) break;
            choice[pos] = 0;
            pos -= 1;
        }
        if (pos < 0) break;
    }
}

inline void enumerate_paths_from_root(
    device atomic_uint* out,
    int word_count,
    const threadgroup ulong* rows,
    const threadgroup uchar* bonds,
    const threadgroup int* labels,
    int atom_count,
    int root,
    int max_length,
    bool query
) {
    int path[8];
    int next_neighbor[8];
    int depth = 0;
    ulong visited = 1UL << root;
    path[0] = root;
    next_neighbor[0] = 0;

    while (depth >= 0) {
        if (depth == max_length || next_neighbor[depth] >= atom_count) {
            visited &= ~(1UL << path[depth]);
            depth -= 1;
            continue;
        }
        int neighbor = next_neighbor[depth];
        next_neighbor[depth] += 1;
        if (((rows[path[depth]] >> neighbor) & 1UL) == 0UL) continue;
        if ((visited & (1UL << neighbor)) != 0UL) continue;

        depth += 1;
        path[depth] = neighbor;
        next_neighbor[depth] = 0;
        visited |= 1UL << neighbor;
        if (path[0] < path[depth]) {
            emit_path_feature_variants(out, word_count, bonds, labels, path, depth, query);
        }
    }
}

struct FingerprintParams {
    uint molecule_count;
    uint query;
    uint max_length;
    uint word_count;
};

kernel void folded_path_fingerprint_kernel(
    device const uint* atoms_global [[buffer(0)]],
    device const uchar* bonds_global [[buffer(1)]],
    device const ulong* rows_global [[buffer(2)]],
    device const uint* atom_counts [[buffer(3)]],
    device atomic_uint* out [[buffer(4)]],
    constant FingerprintParams& params [[buffer(5)]],
    uint blockIdx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint blockDim [[threads_per_threadgroup]]) {

    int mol_idx = (int)blockIdx;
    if (mol_idx >= (int)params.molecule_count) return;

    int atom_count = (int)atom_counts[mol_idx];
    device const uint* atoms = atoms_global + (size_t)mol_idx * BUCKET_SIZE * ATOM_WORD_COUNT;
    device const uchar* bonds_flat = bonds_global + (size_t)mol_idx * BUCKET_SIZE * BUCKET_SIZE;
    device const ulong* rows_flat = rows_global + (size_t)mol_idx * BUCKET_SIZE;
    device atomic_uint* mol_out = out + (size_t)mol_idx * params.word_count * 2;

    threadgroup uint atom_words[BUCKET_SIZE * ATOM_WORD_COUNT];
    threadgroup uchar bonds[BUCKET_SIZE * BUCKET_SIZE];
    threadgroup ulong rows[BUCKET_SIZE];
    threadgroup int labels[BUCKET_SIZE];

    for (uint idx = tid; idx < BUCKET_SIZE * ATOM_WORD_COUNT; idx += blockDim) {
        atom_words[idx] = atoms[idx];
    }
    for (uint idx = tid; idx < BUCKET_SIZE * BUCKET_SIZE; idx += blockDim) {
        bonds[idx] = bonds_flat[idx];
    }
    for (uint row = tid; row < BUCKET_SIZE; row += blockDim) {
        rows[row] = rows_flat[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = tid; row < BUCKET_SIZE; row += blockDim) {
        if ((int)row < atom_count) {
            uint aw[14];
            for (uint i = 0; i < ATOM_WORD_COUNT; ++i) {
                aw[i] = atom_words[row * ATOM_WORD_COUNT + i];
            }
            labels[row] = exact_single_z_from_atom(aw);
        } else {
            labels[row] = -1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int atom = (int)tid; atom < atom_count; atom += (int)blockDim) {
        if (labels[atom] >= 0) {
            set_fingerprint_bit(mol_out, params.word_count, atom_feature_linear(labels[atom]));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int bounded_max_length = (int)params.max_length > 7 ? 7 : (int)params.max_length;
    for (int root = (int)tid; root < atom_count; root += (int)blockDim) {
        enumerate_paths_from_root(mol_out, params.word_count, rows, bonds, labels, atom_count, root, bounded_max_length, params.query != 0);
    }
}

struct PrefilterParams {
    uint molecule_count;
    uint pattern_count;
    uint word_count;
    uint pattern_index_offset;
};

kernel void compact_folded_prefilter_pairs_kernel(
    device const ulong* molecule_fps [[buffer(0)]],
    device const ulong* pattern_fps [[buffer(1)]],
    device const int* molecule_atom_counts [[buffer(2)]],
    device const int* pattern_atom_counts [[buffer(3)]],
    device int* pair_pattern_index [[buffer(4)]],
    device int* pair_molecule_index [[buffer(5)]],
    device atomic_uint* survivor_count [[buffer(6)]],
    constant PrefilterParams& params [[buffer(7)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 blockDim3 [[threads_per_threadgroup]]) {

    uint tid = tid3.x;
    uint blockDim = blockDim3.x;

    threadgroup ulong s_molecules[FP_TILE_SIZE * FP_WORD_COUNT];
    threadgroup ulong s_patterns[FP_TILE_SIZE * FP_WORD_COUNT];
    threadgroup int s_molecule_atom_counts[FP_TILE_SIZE];
    threadgroup int s_pattern_atom_counts[FP_TILE_SIZE];

    int lane = (int)tid % 32;
    int warp_in_block = (int)(tid / 32);
    int warps_per_block = (int)(blockDim / 32);

    int molecule_base = (int)blockIdx.x * FP_TILE_SIZE;
    int pattern_base = (int)blockIdx.y * FP_TILE_SIZE;

    for (int idx = (int)tid; idx < (int)(FP_TILE_SIZE * FP_WORD_COUNT); idx += (int)blockDim) {
        int slot = idx / FP_WORD_COUNT;
        int word = idx % FP_WORD_COUNT;
        int molecule_idx = molecule_base + slot;
        int pattern_idx = pattern_base + slot;
        s_molecules[slot * FP_WORD_COUNT + word] = (molecule_idx < (int)params.molecule_count)
            ? molecule_fps[(size_t)molecule_idx * params.word_count + word]
            : 0UL;
        s_patterns[slot * FP_WORD_COUNT + word] = (pattern_idx < (int)params.pattern_count)
            ? pattern_fps[(size_t)pattern_idx * params.word_count + word]
            : 0UL;
    }
    for (int slot = (int)tid; slot < (int)FP_TILE_SIZE; slot += (int)blockDim) {
        int molecule_idx = molecule_base + slot;
        int pattern_idx = pattern_base + slot;
        s_molecule_atom_counts[slot] = (molecule_idx < (int)params.molecule_count)
            ? (int)molecule_atom_counts[molecule_idx]
            : -1;
        s_pattern_atom_counts[slot] = (pattern_idx < (int)params.pattern_count)
            ? (int)pattern_atom_counts[pattern_idx]
            : 2147483647;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int pair_idx = warp_in_block; pair_idx < (int)(FP_TILE_SIZE * FP_TILE_SIZE); pair_idx += warps_per_block) {
        int molecule_slot = pair_idx % FP_TILE_SIZE;
        int pattern_slot = pair_idx / FP_TILE_SIZE;
        int molecule_idx = molecule_base + molecule_slot;
        int pattern_idx = pattern_base + pattern_slot;
        
        bool local_match =
            molecule_idx < (int)params.molecule_count &&
            pattern_idx < (int)params.pattern_count &&
            s_pattern_atom_counts[pattern_slot] <= s_molecule_atom_counts[molecule_slot];
            
        if (local_match) {
            ulong pattern_word = s_patterns[pattern_slot * FP_WORD_COUNT + lane];
            ulong molecule_word = s_molecules[molecule_slot * FP_WORD_COUNT + lane];
            local_match = (molecule_word & pattern_word) == pattern_word;
        }
        
        bool all_match = simd_all(local_match);
        if (lane == 0 && all_match) {
            uint out_idx = atomic_fetch_add_explicit(survivor_count, 1U, memory_order_relaxed);
            pair_pattern_index[out_idx] = params.pattern_index_offset + pattern_idx;
            pair_molecule_index[out_idx] = molecule_idx;
        }
    }
}
// Reports the compile-time constants this shader was built with; see the same kernel in
// `metal_retro.mm` for why asking beats a comment. BOND_CLASS_MASK rides along because the two
// shaders mask bond flags identically and nothing else checks that they still agree.
kernel void constants_probe(
    device uint* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    out[0] = BUCKET_SIZE;
    out[1] = ATOM_WORD_COUNT;
    out[2] = FP_WORD_COUNT;
    out[3] = FP_TILE_SIZE;
    out[4] = (uint)BOND_CLASS_MASK;
}

)";

// One device, one queue, one shader compilation for this translation unit.
//
// Both entries below used to carry a `dispatch_once` block of their own, so a process that called
// each once made two `MTLDevice`s, two command queues, and compiled this file's whole shader
// source twice. Nothing wanted that: the two kernels come out of the same source string and the
// entries never run concurrently on purpose. Shaped after `metal_retro.mm`'s
// `metal_retro_context`, which is the pattern for the package.
static id<MTLDevice> s_device = nil;
static id<MTLComputePipelineState> s_fingerprint_pipeline = nil;
static id<MTLComputePipelineState> s_prefilter_pipeline = nil;
static id<MTLCommandQueue> s_queue = nil;
static dispatch_once_t s_once;
static bool s_init_failed = false;

static bool metal_pathfp_context(void) {
    dispatch_once(&s_once, ^{
        @autoreleasepool {
            s_device = MTLCreateSystemDefaultDevice();
            if (!s_device) {
                snprintf(g_last_error, sizeof(g_last_error), "No Metal device found");
                s_init_failed = true;
                return;
            }

            NSError* error = nil;
            id<MTLLibrary> library = [s_device newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource] options:nil error:&error];
            if (!library) {
                snprintf(g_last_error, sizeof(g_last_error), "Shader compilation failed: %s", [[error localizedDescription] UTF8String]);
                s_init_failed = true;
                return;
            }

            // Returns nil and sets the error string, so each kernel is one named line below
            // rather than its own copy of the lookup-and-check. A table of slots would be
            // tidier still but ARC rejects a pointer to an `id` with no explicit ownership.
            id<MTLComputePipelineState> (^pipeline_for)(const char*) = ^(const char* name) {
                id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
                if (!function) {
                    snprintf(g_last_error, sizeof(g_last_error), "Function %s not found", name);
                    return (id<MTLComputePipelineState>)nil;
                }
                NSError* pipeline_error = nil;
                id<MTLComputePipelineState> state = [s_device newComputePipelineStateWithFunction:function error:&pipeline_error];
                if (!state) {
                    snprintf(g_last_error, sizeof(g_last_error), "Pipeline creation failed for %s: %s", name, [[pipeline_error localizedDescription] UTF8String]);
                }
                return state;
            };

            s_fingerprint_pipeline = pipeline_for("folded_path_fingerprint_kernel");
            s_prefilter_pipeline = pipeline_for("compact_folded_prefilter_pairs_kernel");
            if (!s_fingerprint_pipeline || !s_prefilter_pipeline) {
                s_init_failed = true;
                return;
            }

            s_queue = [s_device newCommandQueue];
        }
    });
    return !s_init_failed;
}

extern "C" int vesma_metal_linear_fingerprints(
    const uint32_t* atoms,
    const uint8_t* bonds,
    const uint64_t* rows,
    const int32_t* atom_counts,
    uint64_t* out,
    uint32_t molecule_count,
    uint32_t query,
    uint32_t max_length,
    uint32_t word_count) {

    if (!metal_pathfp_context()) return 0;

    @autoreleasepool {
        uint32_t bucket_size = 64;
        uint32_t atom_word_count = 14;
        
        NSUInteger atoms_size = (NSUInteger)molecule_count * bucket_size * atom_word_count * sizeof(uint32_t);
        NSUInteger bonds_size = (NSUInteger)molecule_count * bucket_size * bucket_size * sizeof(uint8_t);
        NSUInteger rows_size = (NSUInteger)molecule_count * bucket_size * sizeof(uint64_t);
        NSUInteger out_size = (NSUInteger)molecule_count * word_count * sizeof(uint64_t);
        
        id<MTLBuffer> buf_atoms = [s_device newBufferWithBytes:atoms length:atoms_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_bonds = [s_device newBufferWithBytes:bonds length:bonds_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_rows = [s_device newBufferWithBytes:rows length:rows_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_atom_counts = [s_device newBufferWithBytes:atom_counts length:molecule_count * sizeof(int32_t) options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> buf_out = [s_device newBufferWithLength:out_size options:MTLResourceStorageModeShared];
        memset([buf_out contents], 0, out_size);

        struct {
            uint32_t molecule_count;
            uint32_t query;
            uint32_t max_length;
            uint32_t word_count;
        } params = {molecule_count, query, max_length, word_count};
        
        id<MTLBuffer> buf_params = [s_device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> commandBuffer = [s_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        commandBuffer.label = @"vesma.folded_path_fingerprint";
        encoder.label = @"folded_path_fingerprint";
        [encoder setComputePipelineState:s_fingerprint_pipeline];
        [encoder setBuffer:buf_atoms offset:0 atIndex:0];
        [encoder setBuffer:buf_bonds offset:0 atIndex:1];
        [encoder setBuffer:buf_rows offset:0 atIndex:2];
        [encoder setBuffer:buf_atom_counts offset:0 atIndex:3];
        [encoder setBuffer:buf_out offset:0 atIndex:4];
        [encoder setBuffer:buf_params offset:0 atIndex:5];

        [encoder dispatchThreadgroups:MTLSizeMake(molecule_count, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if ([commandBuffer status] == MTLCommandBufferStatusError) {
            snprintf(g_last_error, sizeof(g_last_error), "Command buffer error: %s", [[[commandBuffer error] localizedDescription] UTF8String]);
            return 0;
        }

        memcpy(out, [buf_out contents], out_size);
        return 1;
    }
}

extern "C" int vesma_metal_linear_prefilter(
    const uint64_t* molecule_fps,
    const uint64_t* pattern_fps,
    const int32_t* molecule_atom_counts,
    const int32_t* pattern_atom_counts,
    int32_t* pair_pattern_index,
    int32_t* pair_molecule_index,
    uint64_t* survivor_count,
    uint32_t molecule_count,
    uint32_t pattern_count,
    uint32_t word_count,
    uint32_t pattern_index_offset) {

    if (!metal_pathfp_context()) return 0;

    @autoreleasepool {
        uint64_t dense_pair_count = (uint64_t)molecule_count * (uint64_t)pattern_count;
        NSUInteger mol_fps_size = (NSUInteger)molecule_count * word_count * sizeof(uint64_t);
        NSUInteger pat_fps_size = (NSUInteger)pattern_count * word_count * sizeof(uint64_t);
        
        id<MTLBuffer> buf_mol_fps = [s_device newBufferWithBytes:molecule_fps length:mol_fps_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_pat_fps = [s_device newBufferWithBytes:pattern_fps length:pat_fps_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_mol_counts = [s_device newBufferWithBytes:molecule_atom_counts length:molecule_count * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_pat_counts = [s_device newBufferWithBytes:pattern_atom_counts length:pattern_count * sizeof(int32_t) options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> buf_pair_p_idx = [s_device newBufferWithLength:(NSUInteger)dense_pair_count * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_pair_m_idx = [s_device newBufferWithLength:(NSUInteger)dense_pair_count * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_survivor = [s_device newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];
        memset([buf_survivor contents], 0, sizeof(uint64_t));

        struct {
            uint32_t molecule_count;
            uint32_t pattern_count;
            uint32_t word_count;
            uint32_t pattern_index_offset;
        } params = {molecule_count, pattern_count, word_count, pattern_index_offset};
        
        id<MTLBuffer> buf_params = [s_device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> commandBuffer = [s_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        commandBuffer.label = @"vesma.compact_folded_prefilter_pairs";
        encoder.label = @"compact_folded_prefilter_pairs";
        [encoder setComputePipelineState:s_prefilter_pipeline];
        [encoder setBuffer:buf_mol_fps offset:0 atIndex:0];
        [encoder setBuffer:buf_pat_fps offset:0 atIndex:1];
        [encoder setBuffer:buf_mol_counts offset:0 atIndex:2];
        [encoder setBuffer:buf_pat_counts offset:0 atIndex:3];
        [encoder setBuffer:buf_pair_p_idx offset:0 atIndex:4];
        [encoder setBuffer:buf_pair_m_idx offset:0 atIndex:5];
        [encoder setBuffer:buf_survivor offset:0 atIndex:6];
        [encoder setBuffer:buf_params offset:0 atIndex:7];

        uint32_t tile_size = 32;
        uint32_t blocks_x = (molecule_count + tile_size - 1) / tile_size;
        uint32_t blocks_y = (pattern_count + tile_size - 1) / tile_size;

        [encoder dispatchThreadgroups:MTLSizeMake(blocks_x, blocks_y, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if ([commandBuffer status] == MTLCommandBufferStatusError) {
            snprintf(g_last_error, sizeof(g_last_error), "Command buffer error: %s", [[[commandBuffer error] localizedDescription] UTF8String]);
            return 0;
        }

        uint64_t count = *(uint64_t*)[buf_survivor contents];
        *survivor_count = count;
        memcpy(pair_pattern_index, [buf_pair_p_idx contents], (NSUInteger)count * sizeof(int32_t));
        memcpy(pair_molecule_index, [buf_pair_m_idx contents], (NSUInteger)count * sizeof(int32_t));

        return 1;
    }
}

extern "C" int vesma_metal_linear_constants(uint32_t* out) {
    if (!metal_pathfp_context()) return 0;
    @autoreleasepool {
        NSError* error = nil;
        id<MTLLibrary> library = [s_device newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource] options:nil error:&error];
        if (!library) {
            snprintf(g_last_error, sizeof(g_last_error), "Shader compilation failed: %s", [[error localizedDescription] UTF8String]);
            return 0;
        }
        // Its own library and its own pipeline rather than the context's: this runs once, in a
        // test, and the context caches only the two production kernels.
        id<MTLFunction> function = [library newFunctionWithName:@"constants_probe"];
        if (!function) {
            snprintf(g_last_error, sizeof(g_last_error), "Function constants_probe not found");
            return 0;
        }
        id<MTLComputePipelineState> pipeline = [s_device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            snprintf(g_last_error, sizeof(g_last_error), "constants probe pipeline failed: %s", [[error localizedDescription] UTF8String]);
            return 0;
        }
        const NSUInteger count = 5;
        id<MTLBuffer> buf = [s_device newBufferWithLength:count * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [s_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        cb.label = @"vesma.pathfp_constants_probe";
        enc.label = @"pathfp_constants_probe";
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        memcpy(out, [buf contents], count * sizeof(uint32_t));
        return 1;
    }
}

extern "C" const char* vesma_metal_linear_last_error(void) {
    return g_last_error;
}
