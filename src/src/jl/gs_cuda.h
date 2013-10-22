#ifndef GPU_MAP_INIT_H
#define GPU_MAP_INIT_H 

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

void fill_flagged_primaries_map(uint* d_flagged_primaries, const uint* flagged_primaries);

void fill_gpu_maps(uint* map_size, uint** d_map_offsets, uint** d_map_indices_from, uint** d_map_indices_to, const uint* map);

void local_gather_cuda(double* out, const double* in,
                       const uint *map_offsets, const uint* map_indices_from, const uint* map_indices_to, int map_size);

void local_scatter_cuda(double* out, const double* in,
                       const uint *map_offsets, const uint* map_indices_from, const uint* map_indices_to, int map_size);

#ifdef __cplusplus
}
#endif

#endif
