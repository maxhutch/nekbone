#pragma once

#ifdef __cplusplus
extern "C" {
#endif
void gs_setup_cuda(const uint* map_local_0, const uint* map_local_1, const uint* flagged_primaries);

void gs_comm_setup_cuda(const uint comm_0_n, const uint* comm_0_p, const uint* comm_0_size, const uint comm_0_total,
                        const uint comm_1_n, const uint* comm_1_p, const uint* comm_1_size, const uint comm_1_total,
                        const uint* map_comm_0, const uint* map_comm_1,
                        uint buffer_size,
                        const MPI_Comm* mpi_comm,
                        int comm_id,
                        int comm_np);
#ifdef __cplusplus
}
#endif

