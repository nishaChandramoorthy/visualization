#pragma once

#include<cmath>
#include<cstdio>
#include<cinttypes>
#include<png.h>

template<uint32_t nDim, uint32_t iDimX, uint32_t iDimY>
__global__
void draw(uint32_t * density, const float (*state)[nDim], int nState,
          float x0, float dx, int nx,
          float y0, float dy, int ny)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < nState) {
        float x = state[id][iDimX];
        float y = state[id][iDimY];
        int ix = floor((x - x0) / dx);
        int iy = floor((y - y0) / dy);
        if (ix >= 0 and ix < nx and iy >= 0 and iy < ny) {
            atomicAdd(density + iy * nx + ix, 1);
        }
    }
}

__global__
void cmap(uint32_t * density, float power, float multiplier, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n and density[id] > 0) {
        int d = int(powf(density[id] * multiplier, power));
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


