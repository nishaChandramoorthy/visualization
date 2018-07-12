#include "plykin_ode.h"

int main(int argc, char * argv[])
{
    cudaSetDevice((argc > 1) ? atoi(argv[1]) : 0);
    uint32_t nState = 1000000;
    float (*cpuState)[4] = new float[nState][4];
    for (int i = 0; i < nState; ++i) {
        cpuState[i][0] = rand() / float(RAND_MAX);
        cpuState[i][1] = rand() / float(RAND_MAX);
        cpuState[i][2] = rand() / float(RAND_MAX);
        cpuState[i][3] = 0;
    }
    float (*state)[4];
    cudaMalloc(&state, nState * 4 * sizeof(float));
    cudaMemcpy(state, cpuState, nState * 4 * sizeof(float),
               cudaMemcpyHostToDevice);

    uint32_t * density, nx = 1920, ny = 1080;
    cudaMalloc(&density, nx * ny * sizeof(uint32_t));

    advance<<<ceil(nState/128.), 128>>>(state, 6000, nState);

    float s[2] = {1.0f, 1.0f};
    float u0[4], v0[4];
    for (int i = 0; i < 3; ++i) {
        u0[i] = rand() / float(RAND_MAX);
        v0[i] = rand() / float(RAND_MAX);
    }
    u0[3] = v0[3] = 0;
    stepTan(u0, v0, s, 6000);

    float (*mapped)[2];
    cudaMalloc(&mapped, nState * 2 * sizeof(float));

    uint32_t * cpuDensity = new uint32_t[nx * ny];

    int iFrame = 0;
    for (int i = 0; i < 600; ++i) {
        cudaDeviceSynchronize();

        advance<<<ceil(nState/128.), 128>>>(state, 10, nState);
        stepTan(u0, v0, s, 10);
        printf("%d %f %f %f %f %f %f %f %f\n", iFrame,
               u0[0], u0[1], u0[2], u0[3], v0[0], v0[1], v0[2], v0[3]);
        fflush(stdout);
        {
            cudaMemcpyToSymbol(camerau, u0, 3 * sizeof(float));
            cudaMemcpyToSymbol(camerav, v0, 3 * sizeof(float));
            map<<<ceil(nx*ny/128.), 128>>>(mapped, state, nState);
            cudaMemset(density, 0, nx * ny * sizeof(uint32_t));
    
            const double PI = atan2(1.0, 0.0) * 2;
            float dx = 1.5 / ny;
            draw<2,0,1><<<ceil(nState/128.), 128>>>(
                    density, mapped, nState, -12./9, dx, nx, -.75, dx, ny);
            cmap<<<ceil(nx*ny/128.), 128>>>(density, .7, 2000, nx * ny);
            cudaMemcpy(cpuDensity, density,
                       nx * ny * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            char fname[256];
            sprintf(fname, "plykin_ode_local_%03d.png", iFrame++);
            writePng(fname, cpuDensity, nx, ny);
        }
    }

    return 0;
}
