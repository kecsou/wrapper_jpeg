#include <nvjpeg.h>
#include <cuda_runtime_api.h>
#include "wrapperjpeg.hpp"
#include "GPU_NVIDIA/gpu_nvidia.hpp"

struct compressedImage *createCompressedImage(
  unsigned char *bufferEncodedImage,
  size_t sizeBufferEncodedImage,
  double encoder_time) {

  if (bufferEncodedImage == NULL) { //Doesnot allow empty compressedImage to be created
    return NULL;
  }

  struct compressedImage *compressedImage = (struct compressedImage *)malloc(sizeof(struct compressedImage));
  if (compressedImage == NULL) {
    return NULL;
  }

  compressedImage->bufferEncodedImage = bufferEncodedImage;
  compressedImage->sizeBufferEncodedImage = sizeBufferEncodedImage;
  compressedImage->encoder_time = encoder_time;

  return compressedImage;
}

void freeCompressedImage(struct compressedImage *compressedImage) {
  if (compressedImage == NULL) {
    return;
  }

  if (compressedImage->bufferEncodedImage) {
    free(compressedImage->bufferEncodedImage);
  }

  free(compressedImage);
}

struct compressedImage *compressImage(unsigned char * inputBuffer, size_t nSize) {
  return compressImageWithNvidiaGPU(inputBuffer, nSize, NVJPEG_OUTPUT_YUV, NVJPEG_INPUT_RGB, 100);
}
