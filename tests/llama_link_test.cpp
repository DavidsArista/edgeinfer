#include <cstdlib>
#include <iostream>

#include "llama.h"

int main() {
    llama_backend_init();
    const char* system_info = llama_print_system_info();
    if (system_info == nullptr || system_info[0] == '\0') {
        std::cerr << "llama_print_system_info returned empty output\n";
        llama_backend_free();
        return EXIT_FAILURE;
    }
    llama_backend_free();
    std::cout << "llama_link_ok\n";
    return EXIT_SUCCESS;
}
