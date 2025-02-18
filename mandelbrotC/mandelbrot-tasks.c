#include <stdio.h>
#include <cairo/cairo.h>
#include <math.h>
#include <omp.h>  // 🔹 Biblioteca OpenMP

#define WIDTH 4096
#define HEIGHT 4096
#define MAX_ITER 3000

void generate_mandelbrot(const char *filename) {
    cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24, WIDTH, HEIGHT);
    cairo_t *cr = cairo_create(surface);

    cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_paint(cr);

    // Definição da área do plano complexo
    double min_real = -2.0, max_real = 1.0;
    double min_imag = -1.5, max_imag = 1.5;

    // 🔹 Inicia a região paralela
    #pragma omp parallel
    {
        // 🔹 Garante que apenas um thread cria as tarefas
        #pragma omp single
        {
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    #pragma omp task firstprivate(x, y)  // 🔹 Cada pixel será processado como uma tarefa independente
                    {
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

                        // 🔹 Mapeamento de cores baseado no número de iterações
                        double t = log(iter + 1) / log(MAX_ITER);
                        double r = 9 * (1 - t) * t * t * t * 255;
                        double g = 15 * (1 - t) * (1 - t) * t * t * 255;
                        double b = 8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255;

                        // 🔹 Protege o acesso ao Cairo (evita condição de corrida)
                        #pragma omp critical
                        {
                            cairo_set_source_rgb(cr, r / 255.0, g / 255.0, b / 255.0);
                            cairo_rectangle(cr, x, y, 1, 1);
                            cairo_fill(cr);
                        }
                    }
                }
            }
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
    double start, end;

    start = omp_get_wtime();  // 🔹 Tempo real de execução

    generate_mandelbrot("mandelbrot.png");

    end = omp_get_wtime();

    printf("Tempo de execução: %.2f segundos\n", end - start);

    return 0;
}
