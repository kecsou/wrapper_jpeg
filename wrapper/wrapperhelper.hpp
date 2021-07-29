#include <nvjpeg.h>
#include <cuda_runtime_api.h>

inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  return nGpuArchCoresPerSM[index - 1].Cores;
}

inline int gpuGetMaxGflopsDeviceId() {
    int current_device = 0, sm_per_multiproc = 0;
    int max_perf_device = 0;
    int device_count = 0;
    int devices_prohibited = 0;

    uint64_t max_compute_perf = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        return -1;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count) {
        int computeMode = -1, major = 0, minor = 0;
        cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device);

        // If this GPU is not running on Compute Mode prohibited,
        // then we can add it to the list
        if (computeMode != cudaComputeModeProhibited) {
            if (major == 9999 && minor == 9999) {
                sm_per_multiproc = 1;
            } else {
                sm_per_multiproc =
                    _ConvertSMVer2Cores(major,  minor);
            }
            int multiProcessorCount = 0, clockRate = 0;
            cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device);
            cudaError_t result = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
            if (result != cudaSuccess) {
                // If cudaDevAttrClockRate attribute is not supported we
                // set clockRate as 1, to consider GPU with most SMs and CUDA Cores.
                if(result == cudaErrorInvalidValue) {
                    clockRate = 1;
                }
                else {
                    return 0;
                }
            }
            uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

            if (compute_perf > max_compute_perf) {
                max_compute_perf = compute_perf;
                max_perf_device = current_device;
            }
        } else {
            devices_prohibited++;
        }

        ++current_device;
    }

    if (devices_prohibited == device_count) {
        return 0;
    }

    return max_perf_device;
}
