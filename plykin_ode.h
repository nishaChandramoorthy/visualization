#pragma once

#include "vis.h"

__host__ __device__ __forceinline__ float rot_freq(float t)
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

__host__ __device__ __forceinline__ float diff_rot_freq(float t)
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

__host__ __device__ __forceinline__ void step(float u[4], float s[2], int n)
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

__host__ __device__ __forceinline__ float logNorm(float v[4])
{
    float norm2 = 0;
    for (int i = 0; i < 4; ++i) {
        norm2 += v[i] * v[i];
    }
    float norm = sqrt(norm2);
    for (int i = 0; i < 4; ++i) {
        v[i] /= norm;
    }
    return log(norm);
}

__host__ __device__ __forceinline__ float logNormDiff(
        float uPlus[4], float uMinus[4], float eps)
{
    float norm2 = 0;
    for (int i = 0; i < 4; ++i) {
        float vi = (uPlus[i] - uMinus[i]) / (2 * eps);
        norm2 += vi * vi;
    }
    float norm = sqrt(norm2);
    for (int i = 0; i < 4; ++i) {
        float ui = (uPlus[i] + uMinus[i]) / 2;
        float vi = (uPlus[i] - uMinus[i]) / (2 * eps) / norm;
        uPlus[i] = ui + vi * eps;
        uMinus[i] = ui - vi * eps;
    }
    return log(norm);
}

__host__ __device__ __forceinline__ float stepTan(
        float u[4], float v[4], float s[2], int n)
{
    float vUnit[4] = {v[0], v[1], v[2], v[3]};
    float tanMag = logNorm(vUnit);
    float eps = 1E-3;
    float uMinus[4], uPlus[4];
    for (int i = 0; i < 4; ++i) {
        uPlus[i] = u[i] + eps * vUnit[i];
        uMinus[i] = u[i] - eps * vUnit[i];
    }
    for (int i = 0; i < n; ++i) {
        step(uPlus, s, 1);
        step(uMinus, s, 1);
        tanMag += logNormDiff(uPlus, uMinus, eps);
    }
    for (int i = 0; i < 4; ++i) {
        u[i] = (uPlus[i] + uMinus[i]) / 2;
        v[i] = (uPlus[i] - uMinus[i]) / eps;
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

__device__ void rotate(float u[3], const float uRef[3], int i0, int i1)
{
    float r = sqrt(uRef[i0] * uRef[i0] + uRef[i1] * uRef[i1]);
    float a0 = uRef[i0] / r, a1 = uRef[i1] / r;
    float u0 = -a0 * u[i1] + a1 * u[i0];
    float u1 =  a0 * u[i0] + a1 * u[i1];
    u[i0] = u0;
    u[i1] = u1;
}

__constant__ float camerau[3], camerav[3];

__global__
void map(float (*mapped)[2], const float (*state)[4], int nState)
{
    const double PI = atan2(1.0, 0.0) * 2;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float u[3] = {camerau[0], camerau[1], camerau[2]};
    float v[3] = {camerav[0], camerav[1], camerav[2]};
    if (id < nState) {
        float s[3] = {state[id][0], state[id][1], state[id][2]};
        if (abs(u[0]) > abs(u[1])) {
            rotate(s, u, 0, 2);
            rotate(v, u, 0, 2);
            rotate(u, u, 0, 2);
            rotate(s, u, 1, 2);
            rotate(v, u, 1, 2);
            rotate(s, v, 1, 0);
        }
        else {
            rotate(s, u, 1, 2);
            rotate(v, u, 1, 2);
            rotate(u, u, 1, 2);
            rotate(s, u, 0, 2);
            rotate(v, u, 0, 2);
            rotate(s, v, 1, 0);
        }
        mapped[id][0] = s[0];
        mapped[id][1] = s[1];
        if (s[2] < 0) {
            mapped[id][0] = 200 - s[0];
        }
    }
}
