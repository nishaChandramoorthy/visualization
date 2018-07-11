#include "vis.h"

__device__ __forceinline__ void halfstep(float *u, float *s, float sigma,
				float a)
{
	const double PI = atan2(1.0, 0.0) * 2;
	const double T = 6.0;
	double x = u[0];
	double y = u[1];
	double z = u[2];
	double r2 = x*x + y*y + z*z;
	double r = sqrt(r2);
	double rxy2 = x*x + y*y;
	double rxy = sqrt(rxy2);
	double em2erxy2 = exp(-2.0*s[0]*rxy2);
	double emerxy2 = exp(-s[0]*rxy2);
	double term = PI*0.5*(z*sqrt(2.0) + 1.0);
	double sterm = sin(term);
	double cterm = cos(term);
	double emmu = exp(-s[1]);
	double coeff1 = 1.0/((1.0 - emmu)*r + emmu);
    double coeff2 = rxy/sqrt((x*x)*em2erxy2 + 
            (y*y));
    
    u[0] = coeff1*a*z;
    u[1] = coeff1*coeff2*(sigma*x*emerxy2*sterm + 
            y*cterm);
    u[2] = coeff1*coeff2*(-a*x*emerxy2*cterm + 
            a*sigma*y*sterm);
    u[3] = fmod((u[3] + T/2.0),T);
}

__device__ __forceinline__ void step(float u[4], float s[2], int n)
{
	const double PI = atan2(1.0, 0.0) * 2;
	const float c1 = -1.0, c2 = 1.0;
	
    for (int i = 0; i < n; ++i) {
   		halfstep(u,s,c1,c1);
		halfstep(u,s,c2,c2);
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
void map(float (*mapped)[2], const float (*state)[4], int nState)
{
    const double PI = atan2(1.0, 0.0) * 2;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < nState) {
        float x = state[id][0];
        float y = state[id][1];
	    float z = state[id][2];
        mapped[id][0] = x;
        mapped[id][1] = y;
        if (z < 0) {
            mapped[id][0] = 200 - x;
        }
    }
}

int main()
{
    uint32_t nState = 10000000;
    float (*cpuState)[4] = new float[nState][4];
    for (int i = 0; i < nState; ++i) {
        cpuState[i][0] = rand() / float(RAND_MAX);
        cpuState[i][1] = rand() / float(RAND_MAX);
        cpuState[i][2] = rand() / float(RAND_MAX);
        cpuState[i][3] = rand() / float(RAND_MAX);
    }
    float (*state)[4];
    cudaMalloc(&state, nState * 4 * sizeof(float));
    cudaMemcpy(state, cpuState, nState * 4 * sizeof(float),
               cudaMemcpyHostToDevice);
    uint32_t * density, nx = 1920, ny = 1080;
    cudaMalloc(&density, nx * ny * sizeof(uint32_t));
    cudaMemset(density, 0, nx * ny * sizeof(uint32_t));

    const double PI = atan2(1.0, 0.0) * 2;
    float dx = 2. / ny;

    advance<<<ceil(nState/128.), 128>>>(state, 100, nState);

    float (*mapped)[2];
    cudaMalloc(&mapped, nState * 2 * sizeof(float));

    for (int i = 0; i < 100; ++i) {
        advance<<<ceil(nState/128.), 128>>>(state, 2, nState);
        map<<<ceil(nx*ny/128.), 128>>>(mapped, state, nState);
    
        draw<2,0,1><<<ceil(nState/128.), 128>>>(
                density, mapped, nState, -16./9, dx, nx, -1, dx, ny);
    }
    cmap<<<ceil(nx*ny/128.), 128>>>(density, .7, 10, nx * ny);

    uint32_t * cpuDensity = new uint32_t[nx * ny];
    cudaMemcpy(cpuDensity, density, nx * ny * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    writePng("plykin_map.png", cpuDensity, nx, ny);
    return 0;
}
