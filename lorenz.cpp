#include<cmath>
#include<cstdio>
#include<cinttypes>
#include<png.h>

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

__global__
void draw(uint32_t * density, const float (*state)[3], int nState,
          float x0, float dx, int nx,
          float y0, float dy, int ny)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < nState) {
        float x = state[id][0];
        float y = state[id][2];
        int ix = floor((x - x0) / dx);
        int iy = floor((y - y0) / dy);
        if (ix >= 0 and ix < nx and iy >= 0 and iy < ny) {
            atomicAdd(density + iy * nx + ix, 1);
        }
    }
}

__global__
void cmap(uint32_t * density, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n and density[id] > 0) {
        int d = int(powf(float(density[id]), 0.5) * 30);
        if (d > 0x2ff) {
            density[id] = 0xffffffff;
        }
        else if (d > 0x1ff) {
            d -= 0x200;
            density[id] = (0xff << 24) + (d << 16) + (0xff << 8) + 0xff;
        }
        else if (d > 0xff) {
            d -= 0x100;
            density[id] = (0xff << 24) + (d << 8) + 0xff;
        }
        else {
            density[id] = (d << 24) + d;
        }
    }
}

void writePng(const char fileName[], uint32_t * ptr, int nx, int ny)
{
	FILE *fp = fopen(fileName, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  	png_infop info_ptr = png_create_info_struct(png_ptr);
	setjmp(png_jmpbuf(png_ptr));
    png_init_io(png_ptr, fp);
  	setjmp(png_jmpbuf(png_ptr));
	png_set_IHDR(png_ptr, info_ptr, nx, ny,
   			     8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
		         PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png_ptr, info_ptr);
    setjmp(png_jmpbuf(png_ptr));
	png_bytep * row_pointers = new png_bytep[ny];
    for (int iy=0; iy < ny; iy++)
        row_pointers[iy] = png_bytep(ptr + (ny - iy - 1) * nx);
  	png_write_image(png_ptr, row_pointers);
	setjmp(png_jmpbuf(png_ptr));
    png_write_end(png_ptr, NULL);
	delete [] row_pointers;
    fclose(fp);
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
        draw<<<ceil(nState/128.), 128>>>(density, state, nState,
                -dx * nx / 2, dx, nx, 10, dx, ny);
        cmap<<<ceil(nx*ny/128.), 128>>>(density, nx * ny);
        cudaMemcpy(cpuDensity, density, nx * ny * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
        writePng("lorenz.png", cpuDensity, nx, ny);
    }
    return 0;
}
