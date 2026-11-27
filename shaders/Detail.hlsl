// SPDX-License-Identifier: GPL-3.0-only
// Sharpening without grain, and smooth gradients, on the SDR output frame before frame
// generation (VEYRA_TEST_DETAIL=1). Replaces RCAS (Rcas.hlsl) when on.
//
// Why: RCAS keeps nearly its full lobe on a flat area, so it sharpens the game's own noise
// like an edge. Offline on 2026-11-27 (numpy port of Rcas.hlsl), RCAS at 0.707 multiplied
// grain of 1.5 levels by 1.62 and turned the 1-level steps of an 8-bit fog gradient into
// 1.4-level steps. Sebastien saw "a kind of grain" and pixelated fog with it.
//
// 1. Cored luma sharpening. detail d = Y - gaussian5x5(Y); only its part above the noise
//    level is boosted: g(d) = d^3 / (d^2 + core^2), so |d| << core (grain) gets almost
//    nothing and |d| >> core (edges, real texture) gets the full amount. This is the
//    "coring" of video peaking circuits, and the rule of adaptive unsharp masking
//    (Polesel, Ramponi, Mathews, 1997/2000: no gain where local activity is under a noise
//    threshold). The result stays within the 3x3 luma range widened by `overshoot` of that
//    range (RCAS keeps the range exactly: no halo). `smooth` removes that part of the
//    sub-threshold detail: a light spatial grain reduction. Only luma changes, so colour
//    noise is not raised; the same delta goes to R, G and B.
// 2. Debanding. Like libplacebo's deband (and f3kdb): average 4 samples at a random
//    distance (up to radius * iteration) and angle around the pixel, take the average
//    only when it is within threshold / iteration of the pixel on every channel. Only
//    near-flat areas change (fog, sky); anything with more contrast keeps its pixel. The
//    random numbers are fixed per pixel, not per frame: frame generation must not see
//    noise that changes from one real frame to the next.
//    The random sample positions already dither the result: on Sebastien's sky (OBS crop,
//    2026-11-27) threshold 2 levels took 43 steps of up to 1.5 levels to none, for 0.45 level
//    of fine noise, and kept 71/88/91 % of 2/3/4-level textures (3 levels: 56/70/80 %).
// 3. Optional triangular dither of +-1 output level before the 8-bit write (also fixed per
//    pixel), as AdaDeband (Tu et al., IEEE SPL 2020) adds noise after smoothing. Off by
//    default: it added 0.15 to 0.2 level of noise on top of the deband for no fewer steps.
//
// One pass: the sharpening delta comes from the 5x5 luma around the pixel (group-shared
// tile), the deband value from 8 bilinear samples of the same frame. They act on different
// places (deband only where flat, sharpening only above the core), so they are summed.

cbuffer Constants : register(b0)
{
    uint width;
    uint height;
    float amount;     // sharpening gain on the cored detail; 0 = no sharpening
    float core;       // noise level, 0..1 code values (2/255 = 2 levels)
    float overshoot;  // allowed overshoot, fraction of the 3x3 luma range
    float smooth;     // 0..1 of the sub-threshold detail removed
    float w0;         // gaussian weights, centre and distance 1, 2 (sum w0 + 2 w1 + 2 w2 = 1)
    float w1;
    float w2;
    float threshold;  // deband threshold, 0..1 code values; 0 = no deband
    float radius;     // deband radius of the first iteration, pixels
    uint iterations;  // deband iterations
    float dither;     // 1 = dither the output, 0 = round (default)
    uint pad0;
    uint pad1;
    uint pad2;
};

Texture2D<float4> source : register(t0);   // SDR output frame (gamma-encoded RGBA8)
RWTexture2D<float4> target : register(u0);
SamplerState linearClamp : register(s1);

static const float3 kLuma = float3(0.2126, 0.7152, 0.0722);
static const uint kTile = 16 + 4;
groupshared float luma[kTile][kTile];

uint lowbias32(uint x, uint y, uint salt)
{
    uint v = x * 0x8DA6B343u + y * 0xD8163841u + salt * 0xCB1AB31Fu;
    v ^= v >> 16; v *= 0x7FEB352Du;
    v ^= v >> 15; v *= 0x846CA68Bu;
    v ^= v >> 16;
    return v;
}

float random01(uint2 p, uint salt)
{
    return float(lowbias32(p.x, p.y, salt) >> 8) * (1.0 / 16777216.0);
}

[numthreads(16, 16, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID, uint thread : SV_GroupIndex)
{
    const int2 last = int2(width - 1, height - 1);
    const int2 origin = int2(group.xy * 16) - 2;
    for (uint i = thread; i < kTile * kTile; i += 256) {
        const int2 at = clamp(origin + int2(i % kTile, i / kTile), int2(0, 0), last);
        luma[i / kTile][i % kTile] = dot(source[at].rgb, kLuma);
    }
    GroupMemoryBarrierWithGroupSync();

    const uint2 p = group.xy * 16 + local.xy;
    if (p.x >= width || p.y >= height) return;
    const float4 centre = source[p];
    const uint2 t = local.xy + 2;

    // 1. Cored sharpening of luma.
    float delta = 0;
    if (amount > 0 || smooth > 0) {
        const float wk[5] = { w2, w1, w0, w1, w2 };
        float blur = 0;
        [unroll] for (int j = 0; j < 5; ++j) {
            float row = 0;
            [unroll] for (int k = 0; k < 5; ++k) row += wk[k] * luma[t.y + j - 2][t.x + k - 2];
            blur += wk[j] * row;
        }
        const float y = luma[t.y][t.x];
        float lo = y, hi = y;
        [unroll] for (int j = -1; j <= 1; ++j)
            [unroll] for (int k = -1; k <= 1; ++k) {
                const float v = luma[t.y + j][t.x + k];
                lo = min(lo, v); hi = max(hi, v);
            }
        const float d = y - blur;
        const float g = d * d * d / (d * d + core * core + 1e-12);
        const float range = hi - lo;
        const float sharpened = clamp(y + amount * g, lo - overshoot * range, hi + overshoot * range);
        delta = sharpened - smooth * (d - g) - y;
    }

    // 2. Deband.
    float3 colour = centre.rgb;
    if (threshold > 0) {
        const float2 texel = 1.0 / float2(width, height);
        const float2 here = float2(p) + 0.5;
        for (uint it = 1; it <= iterations; ++it) {
            const float angle = random01(p, 2 * it) * 6.28318531;
            const float dist = random01(p, 2 * it + 1) * radius * it;
            float s, c;
            sincos(angle, s, c);
            const float2 a = float2(c, s) * dist, b = float2(-a.y, a.x);
            const float3 average = (source.SampleLevel(linearClamp, (here + a) * texel, 0).rgb +
                                    source.SampleLevel(linearClamp, (here - a) * texel, 0).rgb +
                                    source.SampleLevel(linearClamp, (here + b) * texel, 0).rgb +
                                    source.SampleLevel(linearClamp, (here - b) * texel, 0).rgb) * 0.25;
            const float3 gap = abs(colour - average);
            if (max(gap.r, max(gap.g, gap.b)) < threshold / it) colour = average;
        }
    }

    float3 result = colour + delta;
    // 3. Triangular dither, +-1 level, fixed per pixel.
    if (dither > 0) result += (random01(p, 101) + random01(p, 103) - 1.0) / 255.0;
    target[p] = float4(saturate(result), centre.a);
}
