#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// STB Image Write - biblioteca header-only mais leve que o Cairo
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define WIDTH 16000
#define HEIGHT 16000
#define MAX_ITER 1000
#define NUM_STREAMS 4
#define TILE_SIZE 32
#define BLOCK_SIZE 256  // Otimizado para melhor ocupância em GPUs modernas

// Kernel otimizado usando template para permitir otimizações em tempo de compilação
template<int MAX_ITERATIONS>
__global__ void mandelbrot_kernel(unsigned char* pixels, 
                                const double min_real, const double max_real, 
                                const double min_imag, const double max_imag, 
                                const int width, const int height,
                                const int start_y) {
    // Cache em memória compartilhada para coordenadas
    __shared__ double shared_real[TILE_SIZE];
    __shared__ double shared_imag[TILE_SIZE];
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = start_y + blockIdx.y * blockDim.y + threadIdx.y;
    
    // Precompute e cache das coordenadas
    if (threadIdx.y == 0 && idx < width) {
        shared_real[threadIdx.x] = min_real + idx * ((max_real - min_real) / width);
    }
    if (threadIdx.x == 0 && idy < height) {
        shared_imag[threadIdx.y] = min_imag + idy * ((max_imag - min_imag) / height);
    }
    __syncthreads();
    
    if (idx >= width || idy >= height) return;

    // Otimização: usar registradores para computação intensiva
    const double cr = shared_real[threadIdx.x];
    const double ci = shared_imag[threadIdx.y];
    double zr = 0.0;
    double zi = 0.0;
    int iter = 0;

    // Loop principal otimizado com early exit
    #pragma unroll 8
    while (iter < MAX_ITERATIONS) {
        const double zr2 = zr * zr;
        const double zi2 = zi * zi;
        
        if (zr2 + zi2 > 4.0f) break;
        
        zi = 2.0f * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        iter++;
    }

    // Cálculo de cor otimizado
    const float t = iter < MAX_ITERATIONS ? 
        __logf(__int2float_rd(iter + 1)) * __frcp_rn(__logf(__int2float_rd(MAX_ITERATIONS))) : 
        1.0f;
    
    // Escrita direta na memória global com acesso coalescido
    const int pixel_index = (idy * width + idx) * 3; // Mudamos para RGB (3 bytes) ao invés de RGBA
    pixels[pixel_index] = (unsigned char)(9.0f * (1.0f - t) * t * t * t * 255.0f);        // R
    pixels[pixel_index + 1] = (unsigned char)(15.0f * (1.0f - t) * (1.0f - t) * t * t * 255.0f);  // G
    pixels[pixel_index + 2] = (unsigned char)(8.5f * (1.0f - t) * (1.0f - t) * (1.0f - t) * t * 255.0f);  // B
}

void generate_mandelbrot(const char *filename) {
    // Configuração inicial da GPU
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, mandelbrot_kernel<MAX_ITER>);
    int max_threads_per_block = attr.maxThreadsPerBlock;
    
    // Criar streams com prioridade
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, -i);
    }

    // Alocação de memória otimizada
    const size_t image_size = WIDTH * HEIGHT * 3; // RGB
    unsigned char *h_pixels;
    cudaMallocHost(&h_pixels, image_size);  // Pinned memory
    
    // Prefetch hint para o driver
    cudaMemAdvise(h_pixels, image_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    
    unsigned char *d_pixels;
    cudaMalloc(&d_pixels, image_size);
    cudaMemAdvise(d_pixels, image_size, cudaMemAdviseSetPreferredLocation, 0);
    cudaMemAdvise(d_pixels, image_size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);

    // Configuração de grid otimizada
    const dim3 block_size(TILE_SIZE, TILE_SIZE);
    const dim3 grid_size(
        (WIDTH + block_size.x - 1) / block_size.x,
        (HEIGHT / NUM_STREAMS + block_size.y - 1) / block_size.y
    );

    const int segment_height = HEIGHT / NUM_STREAMS;
    const double min_real = -2.0, max_real = 1.0;
    const double min_imag = -1.5, max_imag = 1.5;

    // Launch kernels em streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        const int start_y = i * segment_height;
        
        mandelbrot_kernel<MAX_ITER><<<grid_size, block_size, 0, streams[i]>>>(
            d_pixels, min_real, max_real, min_imag, max_imag, 
            WIDTH, HEIGHT, start_y
        );

        const size_t segment_size = WIDTH * segment_height * 3;
        cudaMemcpyAsync(
            h_pixels + (start_y * WIDTH * 3),
            d_pixels + (start_y * WIDTH * 3),
            segment_size,
            cudaMemcpyDeviceToHost,
            streams[i]
        );
    }

    // Sincronização e salvamento da imagem
    cudaDeviceSynchronize();
    
    // Usar stb_image_write ao invés do Cairo
    stbi_write_png(filename, WIDTH, HEIGHT, 3, h_pixels, WIDTH * 3);

    // Limpeza
    cudaFreeHost(h_pixels);
    cudaFree(d_pixels);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

int main() {
    // Configurações de performance
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    // Medição de tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    generate_mandelbrot("mandelbrot_final.png");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Tempo de execução: %.2f segundos\n", milliseconds / 1000.0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}