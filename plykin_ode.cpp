#include "vis.h"

__device__ __forceinline__ float rot_freq(float t)
{
    float c0 = 2.0;
    float c1 = 3.0;
    float c2 = 5.0;
    float c3 = 6.0;

    if (c0 <= t and t < c1) return -1;
    if (c2 <= t and t < c3) return +1;
    return 0;

    /*
    float c4 = 0.0;

    float a0 = -1.0;
    float a1 = 0.0;
    float a2 = 1.0;

    float slope = 10.0;
    float est = exp(slope*t);
    float esc0 = exp(slope*c0);
    float esc1 = exp(slope*c1);
    float esc2 = exp(slope*c2);
    float esc3 = exp(slope*c3);
    float esc4 = exp(slope*c4);

    float fn0 = (a1*esc0 + a0*est)/(esc0 + est);	
    float fn1 = (a0*esc1 + a1*est)/(esc1 + est);
    float fn2 = (a1*esc2 + a2*est)/(esc2 + est);
    float fn3 = (a2*esc3 + a1*est)/(esc3 + est);
    float fn4 = (a2*esc4 + a1*est)/(esc4 + est);

    return fn0 + fn1 + fn2 + fn3 + fn4;
    */
}

__device__ __forceinline__ float diff_rot_freq(float t)
{
    float c0 = 1.0;
    float c1 = 2.0;
    float c2 = 4.0;
    float c3 = 5.0;

    if (c0 <= t and t < c1) return -1;
    if (c2 <= t and t < c3) return +1;
    return 0;

    /*
    float a0 = -1.0;
    float a1 = 0.0;
    float a2 = 1.0;

    float slope = 10.0;
    float est = exp(slope*t);
    float esc0 = exp(slope*c0);
    float esc1 = exp(slope*c1);
    float esc2 = exp(slope*c2);
    float esc3 = exp(slope*c3);

    float fn0 = (a1*esc0 + a0*est)/(esc0 + est);	
    float fn1 = (a0*esc1 + a1*est)/(esc1 + est);
    float fn2 = (a1*esc2 + a2*est)/(esc2 + est);
    float fn3 = (a2*esc3 + a1*est)/(esc3 + est);

    return fn0 + fn1 + fn2 + fn3;
    */
}

__device__ __forceinline__ void step(float u[4], float s[2], int n)
{
	const double PI = atan2(1.0, 0.0) * 2;
	const double dt = 0.2e-2;
	const double T = 6.0;
    for (int i = 0; i < n*5; ++i) {
        float x = u[0];
        float y = u[1];
        float z = u[2];
    	float t = u[3];

        float r2 = x * x + y * y + z * z;
        float r = sqrt(r2);
		float sigma = diff_rot_freq(t);
		float a = rot_freq(t);

		float coeff1 = sigma*PI*0.5*(z*sqrt(2.0) + 1.0);
		float coeff2 = s[0]*(1.0 - sigma*sigma - a*a);
		float coeff3 = s[1]*a*a*(1.0 - r);

		u[0] += dt*(-coeff1*y - coeff2*x*y*y + 0.5*a*PI*z + coeff3*x);
        u[1] += dt*(coeff1*x + coeff2*y*x*x + coeff3*y);
        u[2] += dt*(-0.5*a*PI*x + coeff3*z);
        u[3] = fmod((u[3] + dt),T);
	}
}

__global__
void advance(float (*state)[4], int nsteps, int nState)
{
    float s[2] = {1.0f, 1.0f};
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < nState) {
        float u[4];
        for (int i = 0; i < 4; ++i) u[i] = state[id][i];
        step(u, s, nsteps);
        for (int i = 0; i < 4; ++i) state[id][i] = u[i];
    }
}

__global__
void map(float (*mapped)[2], const float (*state)[4], float angle, int nState)
{
    const double PI = atan2(1.0, 0.0) * 2;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < nState) {
        float x0 = state[id][0];
        float y0 = state[id][1];
	    float z0 = state[id][2];
	    float x = x0 * cos(angle) - y0 * sin(angle);
	    float y = x0 * sin(angle) + y0 * cos(angle);
	    float z = z0;
        mapped[id][0] = x;
        mapped[id][1] = z;
        if (y < 0) {
            mapped[id][0] = 200 - z;
        }
    }
}

int main(int argc, char * argv[])
{
    cudaSetDevice((argc > 1) ? atoi(argv[1]) : 0);
    uint32_t nState = 10000000;
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

    const double PI = atan2(1.0, 0.0) * 2;
    float dx = 2. / ny;

    advance<<<ceil(nState/128.), 128>>>(state, 6000, nState);

    float (*mapped)[2];
    cudaMalloc(&mapped, nState * 2 * sizeof(float));

    for (int i = 0; i < 10; ++i) {
        cudaDeviceSynchronize();
        printf("%d\n", i);

        for (int iFrame = 0; iFrame < nFrames; ++iFrame) {
            advance<<<ceil(nState/128.), 128>>>(state, 10, nState);
            map<<<ceil(nx*ny/128.), 128>>>(mapped, state,
                       iFrame * PI / nFrames, nState);
            draw<2,0,1><<<ceil(nState/128.), 128>>>(
                    density + iFrame * nx * ny,
                    mapped, nState, -16./9, dx, nx, -1, dx, ny);
        }
    }

    uint32_t * cpuDensity = new uint32_t[nx * ny];
    for (int iFrame = 0; iFrame < nFrames; ++iFrame) {
        cmap<<<ceil(nx*ny/128.), 128>>>
            (density + iFrame * nx * ny, .7, 50, nx * ny);
        cudaMemcpy(cpuDensity, density + iFrame * nx * ny,
                   nx * ny * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        char fname[256];
        sprintf(fname, "plykin_ode_%03d.png", iFrame);
        writePng(fname, cpuDensity, nx, ny);
    }

    return 0;
}
