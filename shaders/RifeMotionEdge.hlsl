// SPDX-License-Identifier: GPL-3.0-only
// Motion edges of the RIFE flow at full size (VEYRA_TEST_FLOW_RIFE_EDGE=1), first pass: choose.
// RIFE runs at a quarter of the work size, so its motion edges are about ten quarter pixels
// wide once upsampled: around a character that stays still while the camera turns, the
// character's vector leaks 30 to 40 pixels into the background and the camera's vector into
// the top of the head (measured 2026-11-27 on Sebastien's captures and on composites with a
// known answer). XeSS then moves the hair with the background.
// Here a pixel on such an edge (the vectors 40 pixels away on opposite sides differ) tries
// the vectors from those eight places, the camera vector and the zero vector, and keeps the
// one that matches the previous picture best over a 3x3 window, only when it clearly wins:
// cost <= ratio * own cost - 0.005 and below the match limit. Candidate vectors from the
// surroundings are the classic spatial candidates of block-matching frame-rate converters
// (3D recursive search, de Haan et al., IEEE TCSVT 1993), which also always try the zero
// vector. RifeMotionEdgeFilter then drops switches on uncovered background and lone ones.
// Colours are sampled (bilinear, clamp) at the motion grid, so the flow input can be any size.

cbuffer Constants : register(b0)
{
    uint width;        // motion (work) size
    uint height;
    float radius;      // pixels between a pixel and its spatial candidates
    float edgeBase;    // vectors on opposite sides differ by more than edgeBase + edgeRelative * |own|
    float edgeRelative;
    float ratio;       // a candidate wins when cost <= ratio * own cost - 0.005 ...
    float matchLimit;  // ... and cost < matchLimit (mean absolute RGB difference, 0..1)
    float globalScale; // work pixels per GlobalMotion pixel; 0 = no camera candidate
};

Texture2D<float2> motion : register(t0);        // RIFE motion, current -> previous, work pixels
Texture2D<float4> currentColor : register(t1);  // flow input, this frame
Texture2D<float4> previousColor : register(t2); // flow input, previous frame
Texture2D<uint> globalMotion : register(t3);    // GlobalMotion.hlsl (x = 256: median + 128)
RWTexture2D<float2> chosen : register(u0);
RWTexture2D<uint> flags : register(u1);         // 1 = on a motion edge, 2 = switched
SamplerState linearClamp : register(s1);

static const float2 kRing[8] = {
    float2(1, 0), float2(.7071, .7071), float2(0, 1), float2(-.7071, .7071),
    float2(-1, 0), float2(-.7071, -.7071), float2(0, -1), float2(.7071, -.7071) };

float matchCost(float2 centre, float2 v, float3 here[9], float2 texel)
{
    float e = 0;
    [unroll] for (int j = -1; j <= 1; ++j)
        [unroll] for (int i = -1; i <= 1; ++i) {
            const float3 there = previousColor.SampleLevel(linearClamp, (centre + float2(i, j) + v) * texel, 0).rgb;
            e += dot(abs(here[(j + 1) * 3 + i + 1] - there), float3(1, 1, 1)) / 3;
        }
    return e / 9;
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;
    const int2 p = int2(id.xy);
    const int2 last = int2(width - 1, height - 1);
    const float2 own = motion[p];
    float2 ring[8];
    [unroll] for (int k = 0; k < 8; ++k) ring[k] = motion[clamp(p + int2(round(kRing[k] * radius)), 0, last)];
    float spread = 0;
    [unroll] for (int k = 0; k < 4; ++k) spread = max(spread, length(ring[k] - ring[k + 4]));
    if (spread <= edgeBase + edgeRelative * length(own)) {
        chosen[p] = own;
        flags[p] = 0;
        return;
    }
    const float2 texel = 1.0 / float2(width, height);
    const float2 centre = float2(p) + 0.5;
    float3 here[9];
    [unroll] for (int j = -1; j <= 1; ++j)
        [unroll] for (int i = -1; i <= 1; ++i)
            here[(j + 1) * 3 + i + 1] = currentColor.SampleLevel(linearClamp, (centre + float2(i, j)) * texel, 0).rgb;
    const float ownCost = matchCost(centre, own, here, texel);
    float best = 1e9;
    float2 bestVector = own;
    [unroll] for (int k = 0; k < 8; ++k) {
        const float e = matchCost(centre, ring[k], here, texel);
        if (e < best) { best = e; bestVector = ring[k]; }
    }
    if (globalScale > 0) {
        const float2 g = float2(int(globalMotion[uint2(256, 0)]) - 128, int(globalMotion[uint2(256, 1)]) - 128) * globalScale;
        const float e = matchCost(centre, g, here, texel);
        if (e < best) { best = e; bestVector = g; }
    }
    {
        const float e = matchCost(centre, float2(0, 0), here, texel);
        if (e < best) { best = e; bestVector = float2(0, 0); }
    }
    const bool take = best <= ratio * ownCost - 0.005 && best < matchLimit;
    chosen[p] = take ? bestVector : own;
    flags[p] = take ? 3u : 1u;
}
