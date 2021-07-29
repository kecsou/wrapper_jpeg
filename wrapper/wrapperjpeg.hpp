#include <stdlib.h>
#include <nvjpeg.h>
#include <cuda_runtime_api.h>

#if __cplusplus
  extern "C" {
#endif

  struct compressedImage {
    unsigned char *bufferEncodedImage;
    size_t sizeBufferEncodedImage;
    double encoder_time;
  };

  struct compressedImage *createCompressedImage(
    unsigned char *bufferEncodedImage,
    size_t sizeBufferEncodedImage,
    double encoder_time
  );

  void freeCompressedImage(struct compressedImage *compressedImage);

  struct compressedImage *compressImage(unsigned char * inputBuffer, size_t nSize);
#if __cplusplus
  }
#endif


