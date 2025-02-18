#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <complex.h>
#include <png.h>

#define WIDTH 4096  // 4K
#define HEIGHT 4096 // 4K
#define MAX_ITER 1000

// Função para calcular o número de iterações do conjunto de Mandelbrot
int mandelbrot(double complex c) {
    double complex z = 0;
    int iter = 0;
    while (creal(z) * creal(z) + cimag(z) * cimag(z) <= 4 && iter < MAX_ITER) {
        z = z * z + c;
        iter++;
    }
    return iter;
}

// Função para mapear o número de iterações para uma cor RGB
void get_color(int iter, int *r, int *g, int *b) {
    if (iter == MAX_ITER) {
        *r = *g = *b = 0; // Preto para pontos dentro do conjunto
    } else {
        // Mapeia o número de iterações para um gradiente de cores
        *r = (iter * 5) % 256;
        *g = (iter * 7) % 256;
        *b = (iter * 11) % 256;
    }
}

// Função para salvar a imagem em formato PNG
void save_png(const char *filename, int **image) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Erro ao abrir o arquivo %s para escrita.\n", filename);
        return;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "Erro ao criar a estrutura PNG.\n");
        fclose(fp);
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "Erro ao criar a estrutura de informações PNG.\n");
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return;
    }

    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "Erro durante a escrita da imagem PNG.\n");
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return;
    }

    png_init_io(png, fp);

    png_set_IHDR(
        png,
        info,
        WIDTH, HEIGHT,
        8, // Profundidade de cor (8 bits por canal)
        PNG_COLOR_TYPE_RGB, // Usar RGB em vez de escala de cinza
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    // Aloca memória para as linhas da imagem
    png_bytep row = (png_bytep)malloc(3 * WIDTH * sizeof(png_byte)); // 3 canais (RGB)

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int r, g, b;
            get_color(image[y][x], &r, &g, &b); // Obtém a cor RGB
            row[3 * x] = r;     // Canal R
            row[3 * x + 1] = g; // Canal G
            row[3 * x + 2] = b; // Canal B
        }
        png_write_row(png, row);
    }

    free(row);
    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

int main() {
    // Aloca dinamicamente o array 2D para a imagem
    double start = omp_get_wtime();
    int **image = (int **)malloc(HEIGHT * sizeof(int *));
    for (int i = 0; i < HEIGHT; i++) {
        image[i] = (int *)malloc(WIDTH * sizeof(int));
    }

    double x_min = -2.0, x_max = 1.0;
    double y_min = -1.5, y_max = 1.5;
    double dx = (x_max - x_min) / WIDTH;
    double dy = (y_max - y_min) / HEIGHT;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            double x = x_min + j * dx;
            double y = y_min + i * dy;
            double complex c = x + y * I;
            image[i][j] = mandelbrot(c);
        }
    }

    save_png("mandelbrot_4k_rgb.png", image);

    // Libera a memória alocada
    for (int i = 0; i < HEIGHT; i++) {
        free(image[i]);
    }
    free(image);

    double end = omp_get_wtime();

    printf("Tempo de execução: %.2f segundos\n", end - start);
    return 0;
}