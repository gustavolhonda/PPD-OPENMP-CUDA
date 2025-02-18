#include <cairo/cairo.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    double x, y;
} complex;

complex add(complex a, complex b) {
    complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

complex sqr(complex a) {
    complex c;
    c.x = a.x * a.x - a.y * a.y;
    c.y = 2 * a.x * a.y;
    return c;
}

double mod(complex a) {
    return sqrt(a.x * a.x + a.y * a.y);
}

complex mapPoint(int width, int height, double radius, int x, int y) {
    complex c;
    int l = (width < height) ? width : height;
    
    c.x = 2 * radius * (x - width / 2.0) / l;
    c.y = 2 * radius * (y - height / 2.0) / l;
    
    return c;
}

void juliaSet(int width, int height, complex c, double radius, int n, const char* filename) {
    cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
    cairo_t *cr = cairo_create(surface);
    
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            complex z0 = mapPoint(width, height, radius, x, y);
            complex z1;
            int i;
            
            for (i = 0; i < n; i++) {
                z1 = add(sqr(z0), c);
                if (mod(z1) > radius) {
                    break;
                }
                z0 = z1;
            }
            
            double color = (double)i / n;
            cairo_set_source_rgb(cr, color, color * 0.6, 1 - color);
            cairo_rectangle(cr, x, y, 1, 1);
            cairo_fill(cr);
        }
    }
    
    cairo_surface_write_to_png(surface, filename);
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
}

int main(int argC, char* argV[]) {
    if (argC != 7) {
        printf("Usage: %s <width> <height> <real_part> <imaginary_part> <radius> <iterations>\n", argV[0]);
        return 1;
    }
    
    int width = atoi(argV[1]);
    int height = atoi(argV[2]);
    complex c;
    c.x = atof(argV[3]);
    c.y = atof(argV[4]);
    double radius = atof(argV[5]);
    int iterations = atoi(argV[6]);
    
    juliaSet(width, height, c, radius, iterations, "julia_set.png");
    printf("Image saved as julia_set.png\n");
    
    return 0;
}
