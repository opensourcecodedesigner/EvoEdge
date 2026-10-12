#include "results/neural_controller.h"

#ifdef _WIN32
__declspec(dllexport) float c_act(const float* obs) {
    return compute_neural_action(obs);
}
#else
float c_act(const float* obs) {
    return compute_neural_action(obs);
}
#endif
