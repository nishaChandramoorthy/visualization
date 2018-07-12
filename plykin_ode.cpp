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

    int nFrames = 60;
    uint32_t * density, nx = 1920, ny = 1080;
    cudaMalloc(&density, nFrames * nx * ny * sizeof(uint32_t));
    cudaMemset(density, 0, nFrames * nx * ny * sizeof(uint32_t));

    const float PI = atan2(1.0, 0.0) * 2;
    float dx = 2. / ny;

    advance<<<ceil(nState/128.), 128>>>(state, 6000, nState);

    float (*mapped)[2];
    cudaMalloc(&mapped, nState * 2 * sizeof(float));

    for (int i = 0; i < 10; ++i) {
        cudaDeviceSynchronize();
        printf("%d\n", i);

        for (int iFrame = 0; iFrame < nFrames; ++iFrame) {
            advance<<<ceil(nState/128.), 128>>>(state, 10, nState);
            float u[3] = {cos(iFrame * PI / nFrames),
                         -sin(iFrame * PI / nFrames), 0};
            //float v[3] = {sin(iFrame * PI / nFrames),
            //              cos(iFrame * PI / nFrames), 0};
            float v[3] = {0, 0, 1};
            cudaMemcpyToSymbol(camerau, u, 3 * sizeof(float));
            cudaMemcpyToSymbol(camerav, v, 3 * sizeof(float));
            map<<<ceil(nx*ny/128.), 128>>>(mapped, state, nState);
            draw<2,0,1><<<ceil(nState/128.), 128>>>(
                    density + iFrame * nx * ny,
                    mapped, nState, -16./9, dx, nx, -1, dx, ny);
        }
    }

    uint32_t * cpuDensity = new uint32_t[nx * ny];
    for (int iFrame = 0; iFrame < nFrames; ++iFrame) {
        cmap<<<ceil(nx*ny/128.), 128>>>
            (density + iFrame * nx * ny, .7, 100, nx * ny);
        cudaMemcpy(cpuDensity, density + iFrame * nx * ny,
                   nx * ny * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        char fname[256];
        sprintf(fname, "plykin_ode_%03d.png", iFrame);
        writePng(fname, cpuDensity, nx, ny);
    }

    return 0;
}
