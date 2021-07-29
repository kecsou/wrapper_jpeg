#include <nvjpeg.h>
#include <cuda_runtime_api.h>

#if __cplusplus
  extern "C" {
#endif

struct compressedImage *compressImageWithNvidiaGPU(
  unsigned char *inputBuffer,
  size_t sizeInput,
  nvjpegOutputFormat_t output_format,
  nvjpegInputFormat_t input_format,
  int quality
);

#if __cplusplus
}
#endif
