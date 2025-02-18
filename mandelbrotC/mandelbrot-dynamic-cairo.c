#include <stdio.h>
#include <stdlib.h>
#include <cairo/cairo.h>
#include <math.h>
#include <omp.h>

#define WIDTH 4000
#define HEIGHT 4000
#define MAX_ITER 1000

//CUDA ->
//ABORDAGEM QUE PEGA A MELHOR COMPOSIÇÃO DA GRADE
// NÃO GERAR 16 MILHOES THREADS, PENSAR NA MELHOR COMPOSIÇÃO

//OPENMP -> CARGA DO LOOPING EXTERNO

// Estrutura para armazenar cores RGB
typedef struct {
    unsigned char r, g, b;
} Pixel;

void generate_mandelbrot(const char *filename) {
    // 🔹 Criar um buffer de pixels para armazenar a imagem antes de desenhar
    unsigned char *pixels = (unsigned char *)malloc(WIDTH * HEIGHT * 4); // 4 bytes por pixel (ARGB)

    double min_real = -2.0, max_real = 1.0;
    double min_imag = -1.5, max_imag = 1.5;
    double dx = (max_real - min_real) / WIDTH;
    double dy = (max_imag - min_imag) / HEIGHT;

    // 🔹 Paralelizando o processamento da imagem
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double real = min_real + x * dx;
            double imag = min_imag + y * dy;

            double zr = 0.0, zi = 0.0;
            int iter = 0;

            while (zr * zr + zi * zi <= 4.0 && iter < MAX_ITER) {
                double temp = zr * zr - zi * zi + real;
                zi = 2.0 * zr * zi + imag;
                zr = temp;
                iter++;
            }

            // 🔹 Mapeamento de cores
            unsigned char r, g, b;
            if (iter == MAX_ITER) {
                r = g = b = 0; // Preto
            } else {
                double t = log(iter + 1) / log(MAX_ITER);
                r = (int)(9 * (1 - t) * t * t * t * 255);
                g = (int)(15 * (1 - t) * (1 - t) * t * t * 255);
                b = (int)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
            }

            // 🔹 Escrevendo no buffer de imagem (formato ARGB)
            int pixel_index = (y * WIDTH + x) * 4;
            pixels[pixel_index] = b;      // Canal Azul
            pixels[pixel_index + 1] = g;  // Canal Verde
            pixels[pixel_index + 2] = r;  // Canal Vermelho
            pixels[pixel_index + 3] = 255; // Alpha (transparência, sempre 255)
        }
    }

    // 🔹 Criando a superfície de Cairo diretamente a partir do buffer de pixels
    cairo_surface_t *surface = cairo_image_surface_create_for_data(
        pixels, CAIRO_FORMAT_ARGB32, WIDTH, HEIGHT, WIDTH * 4); 

    // 🔹 Criando o contexto Cairo
    cairo_t *cr = cairo_create(surface);
    cairo_set_source_surface(cr, surface, 0, 0);
    cairo_paint(cr);

    // 🔹 Salvando a imagem como PNG
    cairo_surface_write_to_png(surface, filename);

    // 🔹 Limpeza
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
    free(pixels);

    printf("Imagem gerada: %s\n", filename);
}

int main() {
    double start = omp_get_wtime();

    generate_mandelbrot("mandelbrot_optimized.png");

    double end = omp_get_wtime();
    printf("Tempo de execução: %.2f segundos\n", end - start);

    return 0;
}
