#include <stdio.h>
#include <cairo/cairo.h>
#include <math.h>
#include <time.h>

#define WIDTH 4096
#define HEIGHT 4096
#define MAX_ITER 3000

void generate_mandelbrot(const char *filename) {
    cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24, WIDTH, HEIGHT);
    cairo_t *cr = cairo_create(surface);

    cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_fill(cr);

    // Definição da área do plano complexo
    double min_real = -2.0, max_real = 1.0; 
    double min_imag = -1.5, max_imag = 1.5;

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double real = min_real + (x / (double) WIDTH) * (max_real - min_real);
            double imag = min_imag + (y / (double) HEIGHT) * (max_imag - min_imag);

            double zr = 0.0, zi = 0.0;
            int iter = 0;

            while (zr * zr + zi * zi <= 4.0 && iter < MAX_ITER) {
                double temp = zr * zr - zi * zi + real;
                zi = 2.0 * zr * zi + imag;
                zr = temp;
                iter++;
            }

            // Mapeamento de cores (preto para dentro do conjunto, azul-roxo para fora)
            double t = log(iter + 1) / log(MAX_ITER);
            double r = 9 * (1 - t) * t * t * t * 255;
            double g = 15 * (1 - t) * (1 - t) * t * t * 255;
            double b = 8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255;

            cairo_set_source_rgb(cr, r / 255.0, g / 255.0, b / 255.0);
            cairo_rectangle(cr, x, y, 1, 1);
            cairo_fill(cr);
        }
    }

    // Salvar a imagem como PNG
    cairo_surface_write_to_png(surface, filename);

    // Limpeza
    cairo_destroy(cr);
    cairo_surface_destroy(surface);

    printf("Imagem gerada: %s\n", filename);
}

int main() {
    clock_t start, end;
    double cpu_time_used;

    start = clock();
    generate_mandelbrot("mandelbrot.png");
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Tempo de execução: %.2f segundos\n", cpu_time_used);
    return 0;
}
