// Region vectors for repetitive texture (VEYRA_TEST_FLOW_REGION=1, needs VEYRA_TEST_FLOW_GLOBAL=1,
// FidelityFX vectors only). Runs on the optical-flow grid, before FlowRefine.
// On a periodic pattern (a wooden lattice) the matching cost has several almost equal minima,
// one per period, and block flow picks among them by noise: groups of 8x8 cells get a vector
// that is off by a period or by a fraction of it. Frame generation then tears the pattern in
// patches on the generated frames (OBS recordings 2026-11-26: the patches appear with the raw
// FidelityFX vectors too, so they are not made by the per-pixel steps after this one).
// Television frame-rate converters give all blocks of a repetition region one vector, the
// one that explains the whole region (Nguyen Van Thang and Hyuk-Jae Lee, "A semi-global
// motion estimation of a repetition pattern region for frame interpolation", ICIP 2017:
// +2.59 dB), or charge each candidate its distance to the regional vector (US 10057596).
// Here each cell compares its own vector with two region candidates - the camera's motion
// (GlobalMotion: a lattice is scenery, it moves with the camera) and the median of the 5x5
// cells around - on a 16x16 window, twice the cell, which a period shift no longer fits as
// well as a 3x3 patch does. A region candidate replaces the cell's vector when it explains
// the window well (matchLimit) and about as well as the cell's own (relative and absolute
// margin), and only on textured windows (textureMin): there a wrong vector shows, and a
// smooth face or sky, where the camera's vector matches by accident, keeps its own.
// A character's cell keeps its vector: the camera's vector does not explain the character's
// half of the window. One 64-thread group per cell, 2x2 pixels per thread.
Texture2D<int2> rawFlow : register(t0);          // grid extent, FidelityFX integer pixels
Texture2D<float4> previousColor : register(t1);  // flow extent
Texture2D<float4> currentColor : register(t2);   // flow extent
Texture2D<uint> globalMotion : register(t3);     // GlobalMotion.hlsl: x = 256 holds median + 128
RWTexture2D<int2> region : register(u0);         // grid extent: corrected vectors

cbuffer FlowRegionConstants : register(b0)
{
    uint gridW;
    uint gridH;
    uint flowW;
    uint flowH;
    uint gridSize;
    float matchLimit;   // mean luma difference (0..1) a region candidate must stay under
    float margin;       // absolute margin over the cell's own cost (the relative one is 15 %)
    float textureMin;   // mean luma gradient a window needs before its vector may change
};

float luma(float3 c) { return dot(c, float3(.2126, .7152, .0722)); }

groupshared int2 candidates[3];   // own, camera, 5x5 median
groupshared float sums[4][64];    // own, camera, median costs; texture

// Median of 25 values (insertion sort).
int median25(int v[25])
{
    [unroll] for (int i = 1; i < 25; ++i) {
        const int x = v[i];
        int j = i;
        [loop] while (j > 0 && v[j - 1] > x) { v[j] = v[j - 1]; --j; }
        v[j] = x;
    }
    return v[12];
}

[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint index : SV_GroupIndex)
{
    const int2 cell = int2(group.xy);
    if (cell.x >= int(gridW) || cell.y >= int(gridH)) return;   // whole group: uniform
    const int2 cellLimit = int2(gridW - 1, gridH - 1);
    if (index == 0) {
        candidates[0] = rawFlow[cell];
        candidates[1] = int2(int(globalMotion[uint2(256, 0)]) - 128, int(globalMotion[uint2(256, 1)]) - 128);
        int xs[25], ys[25];
        [unroll] for (int k = 0; k < 25; ++k) {
            const int2 v = rawFlow[clamp(cell + int2(k % 5 - 2, k / 5 - 2), int2(0, 0), cellLimit)];
            xs[k] = v.x; ys[k] = v.y;
        }
        candidates[2] = int2(median25(xs), median25(ys));
    }
    GroupMemoryBarrierWithGroupSync();
    // 16x16 window centred on the cell: thread t covers the 2x2 pixels at (t % 8, t / 8) * 2.
    const int2 limit = int2(flowW - 1, flowH - 1);
    const int2 origin = cell * int(gridSize) + int(gridSize) / 2 - 8 + int2(index % 8, index / 8) * 2;
    float cost[3] = { 0, 0, 0 };
    float gradientSum = 0;
    [unroll] for (int s = 0; s < 4; ++s) {
        const int2 p = clamp(origin + int2(s % 2, s / 2), int2(0, 0), limit);
        const float c = luma(currentColor[p].rgb);
        gradientSum += abs(luma(currentColor[min(p + int2(1, 0), limit)].rgb) - c)
                 + abs(luma(currentColor[min(p + int2(0, 1), limit)].rgb) - c);
        [unroll] for (int k = 0; k < 3; ++k)
            cost[k] += abs(c - luma(previousColor[clamp(p + candidates[k], int2(0, 0), limit)].rgb));
    }
    sums[0][index] = cost[0]; sums[1][index] = cost[1]; sums[2][index] = cost[2]; sums[3][index] = gradientSum;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = 32; stride > 0; stride >>= 1) {
        if (index < stride)
            [unroll] for (int k = 0; k < 4; ++k) sums[k][index] += sums[k][index + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (index != 0) return;
    const float own = sums[0][0] / 256, camera = sums[1][0] / 256, median = sums[2][0] / 256;
    const float gradient = sums[3][0] / 512;
    int2 result = candidates[0];
    if (gradient >= textureMin) {
        // The better region candidate; the camera's on a tie (it does not drift with the errors).
        const bool useMedian = median < camera;
        const float best = useMedian ? median : camera;
        const int2 vector = useMedian ? candidates[2] : candidates[1];
        if (any(vector != candidates[0]) && best <= matchLimit && best <= own * 1.15 + margin) result = vector;
    }
    region[cell] = result;
}
