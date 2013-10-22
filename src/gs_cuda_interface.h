#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void gs_cuda_interface(const int *handle, void *u, const int *dom, const int *op,
         const int *transpose);

#ifdef __cplusplus
}
#endif
