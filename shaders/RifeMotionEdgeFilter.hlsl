// SPDX-License-Identifier: GPL-3.0-only
// Motion edges of the RIFE flow at full size (VEYRA_TEST_FLOW_RIFE_EDGE=1), second pass.
// A switch from RifeMotionEdge is good when the chosen vector c lands, in the chosen field, on
// something that moves like c. When it lands on something else (the still character in front),
// the pixel is background just uncovered behind it: no vector can match there, so RIFE's own
// vector stays (the face side of the character is right with it on 2026-11-27 and must not
// change). Then, on a motion edge:
//  - a good switch stays when at least 15 of the 25 pixels around it switched: a motion edge
//    switches whole bands, a lone switch on texture is noise;
//  - a pixel without a good switch that has at least 10 good switches around it takes the one
//    closest to their mean: it fills the holes left in a band (offline 2026-11-27: head-top
//    error on composites with a known answer 7.3 -> 3.7 pixels).

cbuffer Constants : register(b0)
{
    uint width;
    uint height;
    float edgeBase;      // same test as RifeMotionEdge: |a - b| <= edgeBase + edgeRelative * |c|
    float edgeRelative;
    uint minimumSwitched; // of the 5x5 neighbourhood, to keep a switch
    uint minimumFill;     // good switches around a pixel, to fill it
    uint pad0;
    uint pad1;
};

Texture2D<float2> motion : register(t0);  // RIFE motion
Texture2D<float2> chosen : register(t1);  // RifeMotionEdge
Texture2D<uint> flags : register(t2);     // 1 = on a motion edge, 2 = switched
RWTexture2D<float2> output : register(u0);

bool good(int2 q, int2 last, out float2 c)
{
    c = chosen[q];
    if ((flags[q] & 2u) == 0) return false;
    const float2 at = chosen[clamp(int2(round(float2(q) + c)), 0, last)];
    return length(at - c) <= edgeBase + edgeRelative * length(c);
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;
    const int2 p = int2(id.xy);
    const float2 own = motion[p];
    if ((flags[p] & 1u) == 0) { output[p] = own; return; }
    const int2 last = int2(width - 1, height - 1);
    uint switched = 0, count = 0;
    float2 sum = 0;
    [unroll] for (int j = -2; j <= 2; ++j)
        [unroll] for (int i = -2; i <= 2; ++i) {
            const int2 q = p + int2(i, j);
            if (any(q < 0) || any(q > last)) continue;
            switched += (flags[q] >> 1) & 1u;
            float2 c;
            if (good(q, last, c)) { ++count; sum += c; }
        }
    float2 c;
    if (good(p, last, c) && switched >= minimumSwitched) { output[p] = c; return; }
    if (count < minimumFill) { output[p] = own; return; }
    const float2 mean = sum / count;
    float nearest = 1e30;
    float2 pick = own;
    [unroll] for (int j = -2; j <= 2; ++j)
        [unroll] for (int i = -2; i <= 2; ++i) {
            const int2 q = p + int2(i, j);
            if (any(q < 0) || any(q > last)) continue;
            float2 v;
            if (good(q, last, v) && length(v - mean) < nearest) { nearest = length(v - mean); pick = v; }
        }
    output[p] = pick;
}
