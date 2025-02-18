#include <stdio.h>
#include <stdlib.h>
#include <cairo/cairo.h>
#include <cuda_runtime.h>

#define WIDTH 8000
#define HEIGHT 8000
#define MAX_ITER 3000
#define NUM_STREAMS 4  // Número de streams para processamento paralelo
#define TILE_SIZE 32   // Tamanho do tile para melhor coalescência de memória

// Kernel otimizado com uso de memória compartilhada e tiling
__global__ void mandelbrot_kernel(unsigned char* pixels, double min_real, double max_real, 
                                double min_imag, double max_imag, int width, int height,
                                int start_y) {
    __shared__ float shared_results[TILE_SIZE][TILE_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = start_y + blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;

    // Cálculos do Mandelbrot com valores em registradores para maior velocidade
    double dx = (max_real - min_real) / width;
    double dy = (max_imag - min_imag) / height;
    
    double real = min_real + idx * dx;
    double imag = min_imag + idy * dy;

    // Otimização: uso de registradores para valores frequentemente acessados
    double zr = 0.0;
    double zi = 0.0;
    int iter = 0;

    // Loop principal otimizado
    #pragma unroll 4  // Desenrolamento do loop para melhor performance
    while (iter < MAX_ITER) {
        if (zr * zr + zi * zi > 4.0) break;
        double temp = zr * zr - zi * zi + real;
        zi = 2.0 * zr * zi + imag;
        zr = temp;
        iter++;
    }

    // Cálculo de cores otimizado usando instruções rápidas de CUDA
    float t = iter < MAX_ITER ? __logf(__int2float_rd(iter + 1)) / __logf(__int2float_rd(MAX_ITER)) : 1.0f;
    
    // Armazena resultado na memória compartilhada
    shared_results[threadIdx.y][threadIdx.x] = t;
    __syncthreads();

    // Escrita coalescida na memória global
    if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
        float t = shared_results[threadIdx.y][threadIdx.x];
        int pixel_index = (idy * width + idx) * 4;
        
        pixels[pixel_index] = (unsigned char)(8.5f * (1.0f - t) * (1.0f - t) * (1.0f - t) * t * 255.0f);    // Blue
        pixels[pixel_index + 1] = (unsigned char)(15.0f * (1.0f - t) * (1.0f - t) * t * t * 255.0f);        // Green
        pixels[pixel_index + 2] = (unsigned char)(9.0f * (1.0f - t) * t * t * t * 255.0f);                  // Red
        pixels[pixel_index + 3] = 255;                                                                        // Alpha
    }
}

void generate_mandelbrot(const char *filename) {
    // Configuração de streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Alocação de memória com pinned memory para transferência mais rápida
    size_t image_size = WIDTH * HEIGHT * 4 * sizeof(unsigned char);
    unsigned char *h_pixels;
    cudaMallocHost(&h_pixels, image_size);  // Pinned memory
    
    // Alocação de memória no device
    unsigned char *d_pixels;
    cudaMalloc(&d_pixels, image_size);

    // Configuração de grid e block otimizados para ocupância máxima
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(
        (WIDTH + block_size.x - 1) / block_size.x,
        (HEIGHT / NUM_STREAMS + block_size.y - 1) / block_size.y
    );

    int segment_height = HEIGHT / NUM_STREAMS;
    double min_real = -2.0, max_real = 1.0;
    double min_imag = -1.5, max_imag = 1.5;

    // Processamento em streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int start_y = i * segment_height;
        
        // Lançamento do kernel em streams diferentes
        mandelbrot_kernel<<<grid_size, block_size, 0, streams[i]>>>(
            d_pixels, min_real, max_real, min_imag, max_imag, 
            WIDTH, HEIGHT, start_y
        );

        // Transferência assíncrona de dados
        size_t segment_size = WIDTH * segment_height * 4 * sizeof(unsigned char);
        cudaMemcpyAsync(
            h_pixels + (start_y * WIDTH * 4),
            d_pixels + (start_y * WIDTH * 4),
            segment_size,
            cudaMemcpyDeviceToHost,
            streams[i]
        );
    }

    // Sincronização de todos os streams
    cudaDeviceSynchronize();

    // Criar e salvar imagem
    cairo_surface_t *surface = cairo_image_surface_create_for_data(
        h_pixels, CAIRO_FORMAT_ARGB32, WIDTH, HEIGHT, WIDTH * 4
    );

    cairo_t *cr = cairo_create(surface);
    cairo_set_source_surface(cr, surface, 0, 0);
    cairo_paint(cr);
    cairo_surface_write_to_png(surface, filename);

    // Limpeza
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
    cudaFreeHost(h_pixels);  // Libera pinned memory
    cudaFree(d_pixels);

    // Destruir streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    printf("Imagem gerada: %s\n", filename);
}

int main() {
    // Configuração de cache L1/shared memory
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    // Medição de tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    generate_mandelbrot("mandelbrot_cuda_optimized.png");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Tempo de execução: %.2f segundos\n", milliseconds / 1000.0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}