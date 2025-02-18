#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>

int main() {
    // Inicializar GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Erro: Falha ao inicializar GLFW\n");
        return -1;
    }
    printf("GLFW inicializado com sucesso!\n");
    printf("Versão GLFW: %s\n", glfwGetVersionString());

    // Configurar GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Criar janela
    GLFWwindow* window = glfwCreateWindow(640, 480, "Teste OpenGL", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Erro: Falha ao criar janela GLFW\n");
        glfwTerminate();
        return -1;
    }
    printf("Janela GLFW criada com sucesso!\n");

    glfwMakeContextCurrent(window);

    // Inicializar GLEW
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        fprintf(stderr, "Erro: %s\n", glewGetErrorString(err));
        glfwTerminate();
        return -1;
    }
    printf("GLEW inicializado com sucesso!\n");
    printf("Versão OpenGL: %s\n", glGetString(GL_VERSION));
    printf("Renderizador: %s\n", glGetString(GL_RENDERER));

    // Limpar com cor azul clara
    glClearColor(0.2f, 0.3f, 0.8f, 1.0f);

    // Loop principal por 3 segundos
    double startTime = glfwGetTime();
    while (!glfwWindowShouldClose(window) && (glfwGetTime() - startTime) < 3.0) {
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Limpeza
    glfwTerminate();
    printf("Teste concluído com sucesso!\n");
    return 0;
}