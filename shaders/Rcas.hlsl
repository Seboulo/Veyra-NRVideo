// Robust Contrast-Adaptive Sharpening (RCAS) of AMD FidelityFX Super Resolution 1
// (ffx_fsr1.h, MIT, GPUOpen), with its noise attenuation (FSR_RCAS_DENOISE), on the SDR
// output frame before frame generation (VEYRA_TEST_RCAS=<stops>). The sharpening lobe of a
// 5-tap cross is limited per pixel so the result never leaves the [0,1] range of its
// neighbourhood: no halo overshoot, unlike an unsharp mask. Strength 0 stops is the maximum,
// each stop halves it. Captured frames carry no sub-pixel detail to recover: this raises
// edge contrast only, as RCAS does after FSR 1 upscaling.
Texture2D<float4> source : register(t0);   // SDR output colour (gamma-encoded)
RWTexture2D<float4> target : register(u0);

cbuffer RcasConstants : register(b0)
{
    uint width;
    uint height;
    float sharpness;  // exp2(-stops)
    uint unused0;
    uint unused1;
    uint unused2;
    uint unused3;
    uint unused4;
};

#define RCAS_LIMIT (0.25 - (1.0 / 16.0))

float max3(float a, float b, float c) { return max(a, max(b, c)); }
float min3(float a, float b, float c) { return min(a, min(b, c)); }

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;
    const int2 p = int2(id.xy);
    const int2 last = int2(width - 1, height - 1);
    //    b
    //  d e f
    //    h
    const float4 eFull = source[p];
    const float3 b = source[clamp(p + int2(0, -1), int2(0, 0), last)].rgb;
    const float3 d = source[clamp(p + int2(-1, 0), int2(0, 0), last)].rgb;
    const float3 e = eFull.rgb;
    const float3 f = source[clamp(p + int2(1, 0), int2(0, 0), last)].rgb;
    const float3 h = source[clamp(p + int2(0, 1), int2(0, 0), last)].rgb;
    // Luma times 2.
    const float bL = b.b * .5 + (b.r * .5 + b.g);
    const float dL = d.b * .5 + (d.r * .5 + d.g);
    const float eL = e.b * .5 + (e.r * .5 + e.g);
    const float fL = f.b * .5 + (f.r * .5 + f.g);
    const float hL = h.b * .5 + (h.r * .5 + h.g);
    // Noise detection: a pixel unlike its whole cross is grain, sharpen it less.
    float nz = .25 * (bL + dL + fL + hL) - eL;
    const float range = max(max3(max3(bL, dL, eL), fL, hL) - min3(min3(bL, dL, eL), fL, hL), 1e-5);
    nz = saturate(abs(nz) / range);
    nz = -.5 * nz + 1.0;
    // Min and max of the ring.
    const float3 mn4 = min(min(min(b, d), f), h);
    const float3 mx4 = max(max(max(b, d), f), h);
    // Limiters: the largest negative lobe that keeps the result inside [0,1].
    const float2 peakC = float2(1.0, -4.0);
    const float3 hitMin = min(mn4, e) / (4.0 * mx4 + 1e-5);
    // The denominator is 4*mn4-4 <= 0; keep it off zero for an all-white ring.
    const float3 hitMax = (peakC.x - max(mx4, e)) / min(4.0 * mn4 + peakC.y, -1e-5);
    const float3 lobeRGB = max(-hitMin, hitMax);
    float lobe = max(-RCAS_LIMIT, min(max3(lobeRGB.r, lobeRGB.g, lobeRGB.b), 0.0)) * sharpness;
    lobe *= nz;
    const float rcpL = 1.0 / (4.0 * lobe + 1.0);
    const float3 result = (lobe * (b + d + f + h) + e) * rcpL;
    target[p] = float4(saturate(result), eFull.a);
}
