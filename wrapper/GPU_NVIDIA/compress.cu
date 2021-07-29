#include <stdlib.h>
#include <vector>
#include <nvjpeg.h>
#include <cuda_runtime_api.h>
#include "gpu_nvidia.hpp"
#include "../wrapperjpeg.hpp"
#include "../wrapperhelper.hpp"

int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }
int dev_free(void *p) { return (int)cudaFree(p); }

bool is_interleaved(nvjpegOutputFormat_t format)
{
    if (format == NVJPEG_OUTPUT_RGBI || format == NVJPEG_OUTPUT_BGRI)
        return true;
    else
        return false;
}

struct compressedImage *compressImageWithNvidiaGPU(
    unsigned char *inputBuffer,
    size_t sizeInput,
    nvjpegOutputFormat_t output_format,
    nvjpegInputFormat_t input_format,
    int quality
) {
    cudaEvent_t startEvent = NULL, stopEvent = NULL;

    nvjpegEncoderParams_t encode_params;
    nvjpegHandle_t nvjpeg_handle;
    nvjpegJpegState_t jpeg_state;
    nvjpegEncoderState_t encoder_state;
    nvjpegChromaSubsampling_t subsampling;
    std::vector<unsigned char> obuffer;
    size_t sizeOutputBuffer;

    unsigned char *bufferEncodedImage = NULL;
    float loopTime = 0;
    int huf = 0;
    int dev = gpuGetMaxGflopsDeviceId();
    if (dev == -1) {
        return NULL;
    }

    cudaSetDevice(dev);

    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &nvjpeg_handle);
    nvjpegJpegStateCreate(nvjpeg_handle, &jpeg_state);
    nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, NULL);
    nvjpegEncoderParamsCreate(nvjpeg_handle, &encode_params, NULL);

    nvjpegEncoderParamsSetQuality(encode_params, quality, NULL);
    nvjpegEncoderParamsSetOptimizedHuffman(encode_params, huf, NULL);
    nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_420, NULL);

    // Retrieve the componenet and size info.
    int nComponent = 0;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    if (NVJPEG_STATUS_SUCCESS != nvjpegGetImageInfo(
        nvjpeg_handle,
        inputBuffer,
        sizeInput,
        &nComponent,
        &subsampling,
        widths,
        heights
        )
    ) {
        return NULL;
    }

    if (subsampling == NVJPEG_CSS_UNKNOWN)
    {
        return NULL;
    }

    unsigned char* gpuBuffer = NULL;
    cudaError_t eCopy = cudaMalloc((void**)&gpuBuffer, widths[0] * heights[0] * NVJPEG_MAX_COMPONENT);
    if(cudaSuccess != eCopy) 
    {
        return NULL;
    }

    nvjpegImage_t imgdesc = 
    {
        {
            gpuBuffer,
            gpuBuffer + widths[0]*heights[0],
            gpuBuffer + widths[0]*heights[0]*2,
            gpuBuffer + widths[0]*heights[0]*3
        },
        {
            (unsigned int)(is_interleaved(output_format) ? widths[0] * 3 : widths[0]),
            (unsigned int)widths[0],
            (unsigned int)widths[0],
            (unsigned int)widths[0]
        }
    };

    int nReturnCode = 0;

    cudaDeviceSynchronize();

    nReturnCode = nvjpegDecode(nvjpeg_handle, jpeg_state, inputBuffer, sizeInput, output_format, &imgdesc, NULL);
    cudaDeviceSynchronize();

    if(nReturnCode != 0)
    {
      return NULL;
    }

    cudaEventRecord(startEvent, NULL);

    /////////////////////// encode ////////////////////
    if (NVJPEG_OUTPUT_YUV == output_format)
    {
        nvjpegEncodeYUV(nvjpeg_handle,
            encoder_state,
            encode_params,
            &imgdesc,
            subsampling,
            widths[0],
            heights[0],
            NULL);
    }
    else
    {
        nvjpegEncodeImage(nvjpeg_handle,
            encoder_state,
            encode_params,
            &imgdesc,
            input_format,
            widths[0],
            heights[0],
            NULL);
    }

    nvjpegEncodeRetrieveBitstream(
        nvjpeg_handle,
        encoder_state,
        NULL,
        &sizeOutputBuffer,
        NULL);

    obuffer.resize(sizeOutputBuffer);

    nvjpegEncodeRetrieveBitstream(
        nvjpeg_handle,
        encoder_state,
        obuffer.data(),
        &sizeOutputBuffer,
        NULL);

    cudaEventRecord(stopEvent, NULL);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&loopTime, startEvent, stopEvent);

    bufferEncodedImage = (unsigned char *)malloc(sizeof(unsigned char) * obuffer.size());

    for (unsigned int i = 0; i < obuffer.size(); i++) {
        bufferEncodedImage[i] = obuffer[i];
    }

    cudaFree(gpuBuffer);

    nvjpegEncoderParamsDestroy(encode_params);
    nvjpegEncoderStateDestroy(encoder_state);
    nvjpegJpegStateDestroy(jpeg_state);
    nvjpegDestroy(nvjpeg_handle);

    return createCompressedImage(bufferEncodedImage, obuffer.size(), static_cast<double>(loopTime));
}
