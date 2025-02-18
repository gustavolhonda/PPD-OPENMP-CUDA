#include <stdio.h>
#include <stdlib.h>
#include <cairo/cairo.h>
#include <cuda_runtime.h>

#define WIDTH 8000
#define HEIGHT 8000
#define MAX_ITER 1000

// Kernel for Mandelbrot set computation
__global__ void mandelbrot_kernel(unsigned char* pixels, double min_real, double max_real, 
                                double min_imag, double max_imag, int width, int height) {
    // Calculate thread's global position
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;

    // Calculate pixel position in complex plane
    double dx = (max_real - min_real) / width;
    double dy = (max_imag - min_imag) / height;
    
    double real = min_real + idx * dx;
    double imag = min_imag + idy * dy;

    // Mandelbrot iteration
    double zr = 0.0, zi = 0.0;
    int iter = 0;

    while (zr * zr + zi * zi <= 4.0 && iter < MAX_ITER) {
        double temp = zr * zr - zi * zi + real;
        zi = 2.0 * zr * zi + imag;
        zr = temp;
        iter++;
    }

    // Color mapping using CUDA's built-in logf function
    unsigned char r, g, b;
    if (iter == MAX_ITER) {
        r = g = b = 0;
    } else {
        float t = __logf(__int2float_rd(iter + 1)) / __logf(__int2float_rd(MAX_ITER));
        r = (unsigned char)(9.0f * (1.0f - t) * t * t * t * 255.0f);
        g = (unsigned char)(15.0f * (1.0f - t) * (1.0f - t) * t * t * 255.0f);
        b = (unsigned char)(8.5f * (1.0f - t) * (1.0f - t) * (1.0f - t) * t * 255.0f);
    }

    // Write to global memory (ARGB format)
    int pixel_index = (idy * width + idx) * 4;
    pixels[pixel_index] = b;      // Blue
    pixels[pixel_index + 1] = g;  // Green
    pixels[pixel_index + 2] = r;  // Red
    pixels[pixel_index + 3] = 255;// Alpha
}

void generate_mandelbrot(const char *filename) {
    // Allocate host memory
    size_t image_size = WIDTH * HEIGHT * 4 * sizeof(unsigned char);
    unsigned char *h_pixels = (unsigned char *)malloc(image_size);

    // Allocate device memory
    unsigned char *d_pixels;
    cudaMalloc(&d_pixels, image_size);

    // Define grid and block dimensions
    dim3 block_size(16, 16);  // 256 threads per block
    dim3 grid_size(
        (WIDTH + block_size.x - 1) / block_size.x,
        (HEIGHT + block_size.y - 1) / block_size.y
    );

    // Launch kernel
    double min_real = -2.0, max_real = 1.0;
    double min_imag = -1.5, max_imag = 1.5;

    mandelbrot_kernel<<<grid_size, block_size>>>(
        d_pixels, min_real, max_real, min_imag, max_imag, WIDTH, HEIGHT
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_pixels);
        free(h_pixels);
        return;
    }

    // Copy result back to host
    cudaMemcpy(h_pixels, d_pixels, image_size, cudaMemcpyDeviceToHost);

    // Create Cairo surface and save image
    cairo_surface_t *surface = cairo_image_surface_create_for_data(
        h_pixels, CAIRO_FORMAT_ARGB32, WIDTH, HEIGHT, WIDTH * 4
    );

    cairo_t *cr = cairo_create(surface);
    cairo_set_source_surface(cr, surface, 0, 0);
    cairo_paint(cr);
    cairo_surface_write_to_png(surface, filename);

    // Cleanup
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
    cudaFree(d_pixels);
    free(h_pixels);

    printf("Image generated: %s\n", filename);
}

int main() {
    // Record start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    generate_mandelbrot("mandelbrot_cuda.png");

    // Record end time and calculate duration
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Execution time: %.2f seconds\n", milliseconds / 1000.0);

    // Cleanup timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}