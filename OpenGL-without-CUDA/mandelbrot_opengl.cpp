#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glcorearb.h>

// Vertex shader para renderizar um quad em tela cheia
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

// Fragment shader para calcular o conjunto de Mandelbrot
const char* fragmentShaderSource = R"(
#version 400 core
out vec4 FragColor;
in vec2 TexCoord;

uniform double zoom;
uniform dvec2 center;
uniform int maxIterations;

void main() {
    dvec2 c = center + (dvec2(TexCoord) - dvec2(0.5)) * zoom;
    dvec2 z = dvec2(0.0);
    int iter;
    
    for(iter = 0; iter < maxIterations; iter++) {
        double x = z.x * z.x - z.y * z.y + c.x;
        double y = 2.0 * z.x * z.y + c.y;
        
        if(x*x + y*y > 4.0) break;
        z.x = x;
        z.y = y;
    }
    
    if(iter == maxIterations) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        float t = float(iter) / float(maxIterations);
        FragColor = vec4(
            9.0 * (1.0 - t) * t * t * t,
            15.0 * (1.0 - t) * (1.0 - t) * t * t,
            8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t,
            1.0
        );
    }
}
)";

// Estado global para interação
struct {
    double zoom;
    double centerX;
    double centerY;
    int maxIterations;
    bool isDragging;
    double lastX;
    double lastY;
} state = {
    .zoom = 4.0,
    .centerX = -0.5,
    .centerY = 0.0,
    .maxIterations = 1000,
    .isDragging = false
};

// Callback para mouse
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

// Callback para scroll do mouse (zoom)
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    double zoomFactor = pow(0.9, yoffset);
    state.zoom *= zoomFactor;
}

// Callback para movimento do mouse
void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    if (state.isDragging) {
        double dx = xpos - state.lastX;
        double dy = ypos - state.lastY;
        
        state.centerX -= dx * state.zoom / 800.0;
        state.centerY += dy * state.zoom / 800.0;
        
        state.lastX = xpos;
        state.lastY = ypos;
    }
}

// Compilar shader
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
    // Inicializar GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Falha ao inicializar GLFW\n");
        return -1;
    }

    // Configurar GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Criar janela
    GLFWwindow* window = glfwCreateWindow(800, 800, "Mandelbrot Set - OpenGL", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Falha ao criar janela GLFW\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    
    // Inicializar GLEW
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Falha ao inicializar GLEW\n");
        return -1;
    }

    // Configurar callbacks
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);

    // Compilar e linkar shaders
    GLuint vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
    GLuint fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);
    
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Verificar linkagem
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
        // posições    // coordenadas de textura
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    // Criar e configurar VAO/VBO
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

    // Localizar uniforms
    GLint zoomLoc = glGetUniformLocation(shaderProgram, "zoom");
    GLint centerLoc = glGetUniformLocation(shaderProgram, "center");
    GLint maxIterationsLoc = glGetUniformLocation(shaderProgram, "maxIterations");

    // Loop principal
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);
        
        glUseProgram(shaderProgram);
        
        // Atualizar uniforms
        glUniform1d(zoomLoc, state.zoom);
        glUniform2d(centerLoc, state.centerX, state.centerY);
        glUniform1i(maxIterationsLoc, state.maxIterations);
        
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Limpeza
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    glfwTerminate();
    return 0;
}