#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct MandelbrotParams {
    double zoom;
    double centerX;
    double centerY;
    int maxIterations;
    int width;
    int height;
};

__global__ void mandelbrotKernel(uchar4* output, MandelbrotParams params) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= params.width || y >= params.height) return;
    
    double scale = params.zoom / params.width;
    double cx = params.centerX + (x - params.width/2.0) * scale;
    double cy = params.centerY + (y - params.height/2.0) * scale;
    
    double zx = 0.0;
    double zy = 0.0;
    int iter;
    
    for (iter = 0; iter < params.maxIterations; iter++) {
        double zx2 = zx * zx;
        double zy2 = zy * zy;
        
        if (zx2 + zy2 > 4.0) break;
        
        double temp = zx2 - zy2 + cx;
        zy = 2.0 * zx * zy + cy;
        zx = temp;
    }
    
    uchar4 color;
    if (iter == params.maxIterations) {
        color = make_uchar4(0, 0, 0, 255);
    } else {
        float t = float(iter) / float(params.maxIterations);
        color = make_uchar4(
            (unsigned char)(255 * 9.0 * (1.0 - t) * t * t * t),
            (unsigned char)(255 * 15.0 * (1.0 - t) * (1.0 - t) * t * t),
            (unsigned char)(255 * 8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t),
            255
        );
    }
    
    output[y * params.width + x] = color;
}

// Vertex e Fragment shaders
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D mandelbrotTexture;

void main() {
    FragColor = texture(mandelbrotTexture, TexCoord);
}
)";

struct {
  double zoom;
  double centerX;
  double centerY;
  int maxIterations;
  bool isDragging;
  double lastX;
  double lastY;
  bool shouldUpdate;  // Nova flag para controlar atualizações
} state = {
  .zoom = 4.0,
  .centerX = -0.5,
  .centerY = 0.0,
  .maxIterations = 100,  // Começando com menos iterações
  .isDragging = false,
  .shouldUpdate = true
};

struct {
    GLuint texture;
    uchar4* deviceBuffer;
    uchar4* hostBuffer;  // Buffer na CPU para transferência
    bool initialized;
} resources = {0, NULL, NULL, false};

// Callbacks do mouse (mesmo código anterior)
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
      if (action == GLFW_PRESS) {
          state.isDragging = true;
          glfwGetCursorPos(window, &state.lastX, &state.lastY);
      } else if (action == GLFW_RELEASE) {
          state.isDragging = false;
      }
  }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (action == GLFW_PRESS || action == GLFW_REPEAT) {
      switch (key) {
          case GLFW_KEY_UP:
              state.maxIterations += 100;
              printf("Iterações aumentadas para: %d\n", state.maxIterations);
              state.shouldUpdate = true;
              break;
          case GLFW_KEY_DOWN:
              if (state.maxIterations > 100) {
                  state.maxIterations -= 100;
                  printf("Iterações reduzidas para: %d\n", state.maxIterations);
                  state.shouldUpdate = true;
              }
              break;
      }
  }
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
  double zoomFactor = pow(0.9, yoffset);
  state.zoom *= zoomFactor;
  state.shouldUpdate = true;
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
  if (state.isDragging) {
      double dx = xpos - state.lastX;
      double dy = ypos - state.lastY;
      
      state.centerX -= dx * state.zoom / 800.0;
      state.centerY += dy * state.zoom / 800.0;
      
      state.lastX = xpos;
      state.lastY = ypos;
      state.shouldUpdate = true;
  }
}

void updateMandelbrot(int width, int height) {
    MandelbrotParams params = {
        state.zoom,
        state.centerX,
        state.centerY,
        state.maxIterations,
        width,
        height
    };
    
    dim3 blockSize(16, 16);
    dim3 numBlocks((width + blockSize.x - 1) / blockSize.x,
                   (height + blockSize.y - 1) / blockSize.y);
    
    mandelbrotKernel<<<numBlocks, blockSize>>>(resources.deviceBuffer, params);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copiar resultado para o buffer da CPU
    CHECK_CUDA_ERROR(cudaMemcpy(resources.hostBuffer, resources.deviceBuffer, 
                               width * height * sizeof(uchar4), 
                               cudaMemcpyDeviceToHost));
    
    // Atualizar textura OpenGL
    glBindTexture(GL_TEXTURE_2D, resources.texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL_RGBA, GL_UNSIGNED_BYTE, resources.hostBuffer);
}

GLuint compileShader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        fprintf(stderr, "Shader compilation failed: %s\n", infoLog);
        exit(1);
    }
    
    return shader;
}

int main() {
    // Inicializar CUDA
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    
    // Inicializar GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Falha ao inicializar GLFW\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 800, "Mandelbrot Set - CUDA + OpenGL", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Falha ao criar janela GLFW\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Falha ao inicializar GLEW\n");
        return -1;
    }

    // Configurar callbacks
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);

    // Criar e configurar textura OpenGL
    glGenTextures(1, &resources.texture);
    glBindTexture(GL_TEXTURE_2D, resources.texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 800, 800, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Alocar buffers
    CHECK_CUDA_ERROR(cudaMalloc(&resources.deviceBuffer, 800 * 800 * sizeof(uchar4)));
    resources.hostBuffer = (uchar4*)malloc(800 * 800 * sizeof(uchar4));
    if (!resources.hostBuffer) {
        fprintf(stderr, "Falha ao alocar buffer na CPU\n");
        return -1;
    }
    
    resources.initialized = true;

    // Compilar e linkar shaders
    GLuint vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
    GLuint fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);
    
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        fprintf(stderr, "Shader program linking failed: %s\n", infoLog);
        return -1;
    }

    // Criar quad em tela cheia
    float vertices[] = {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Loop principal
    while (!glfwWindowShouldClose(window)) {
      if (state.shouldUpdate) {
          updateMandelbrot(800, 800);
          state.shouldUpdate = false;
      }
      
      glClear(GL_COLOR_BUFFER_BIT);
      glUseProgram(shaderProgram);
      glBindTexture(GL_TEXTURE_2D, resources.texture);
      glBindVertexArray(VAO);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      
      glfwSwapBuffers(window);
      glfwPollEvents();
  }

    // Limpeza
    CHECK_CUDA_ERROR(cudaFree(resources.deviceBuffer));
    free(resources.hostBuffer);
    glDeleteTextures(1, &resources.texture);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    glfwTerminate();
    return 0;
} 