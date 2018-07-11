#include "vis.h"

__global__
void advance(float (*state)[3], float dt, int nsteps, int nState)
{
    float sigma = 10, beta = 8./3, rho = 28.;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < nState) {
        float x = state[id][0];
        float y = state[id][1];
        float z = state[id][2];
        for (int i = 0; i < nsteps; ++i) {
            float dx = sigma * (y - x);
            float dy = x * (rho - z) - y;
            float dz = x * y - beta * z;
            x += dt * dx;
            y += dt * dy;
            z += dt * dz;
        }
        state[id][0] = x;
        state[id][1] = y;
        state[id][2] = z;
    }
}

int main()
{
    uint32_t nState = 100000000;
    float (*cpuState)[3] = new float[nState][3];
    for (int i = 0; i < nState; ++i) {
        cpuState[i][0] = rand() / float(RAND_MAX);
        cpuState[i][1] = rand() / float(RAND_MAX);
        cpuState[i][2] = rand() / float(RAND_MAX) + 28;
    }
    float (*state)[3];
    cudaMalloc(&state, nState * 3 * sizeof(float));
    cudaMemcpy(state, cpuState, nState * 3 * sizeof(float),
               cudaMemcpyHostToDevice);
    uint32_t * density, * cpuDensity, nx = 1920, ny = 1080;
    cudaMalloc(&density, nx * ny * sizeof(uint32_t));
    cpuDensity = new uint32_t[nx * ny];
    float dx = 0.025;

    for (int i = 0; i < 1; ++i) {
        advance<<<ceil(nState/128.), 128>>>(state, 0.001, 10000, nState);
        cudaMemset(density, 0, nx * ny * sizeof(uint32_t));
        draw<3,0,2><<<ceil(nState/128.), 128>>>(
                density, state, nState, -dx * nx / 2, dx, nx, 10, dx, ny);
        cmap<<<ceil(nx*ny/128.), 128>>>(density, 0.5, 100, nx * ny);
        cudaMemcpy(cpuDensity, density, nx * ny * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
        writePng("lorenz.png", cpuDensity, nx, ny);
    }
    return 0;
}
