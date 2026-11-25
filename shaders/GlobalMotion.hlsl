// Camera (global) motion of the frame for FlowRefine (VEYRA_TEST_FLOW_GLOBAL=1): the median
// of all optical-flow grid vectors, per component. While the camera turns, most of the picture
// is background, so the median is the background's motion even when a character covers a
// large part of the centre. Frame-rate converters in television sets keep such a global
// candidate beside local ones and treat a region that moves unlike it as foreground; XeSS
// frame generation itself chooses per pixel between two motion candidates (game vectors and
// its own flow; Intel XeSS 2 white paper). Three dispatches: clear, accumulate, resolve.
Texture2D<int2> rawFlow : register(t0);          // grid extent: NVOF S10.5 or FidelityFX integer pixels
RWTexture2D<uint> histogram : register(u0);      // 257 x 2: bins 0..255 (vector + 128), x = 256 result

cbuffer GlobalMotionConstants : register(b0)
{
    uint gridW;
    uint gridH;
    uint mode;     // 0 clear, 1 accumulate, 2 resolve; 3 accumulate and 4 resolve the foreground
    float scale;   // raw units to pixels
    uint unused0;
    uint unused1;
    uint unused2;
    uint unused3;
};

groupshared uint localBins[512];

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID, uint index : SV_GroupIndex)
{
    if (mode == 0) {
        if (id.x <= 256 && id.y < 4) histogram[id.xy] = id.x == 256 ? (id.y < 2 ? 128u : 1023u) : 0u;
        return;
    }
    const uint row0 = mode >= 3 ? 2u : 0u;
    if (mode == 2 || mode == 4) {
        // One thread per component walks the cumulative count to the middle sample.
        if (id.x != 0 || id.y >= 2) return;
        const uint row = row0 + id.y;
        uint total = 0;
        for (uint b = 0; b < 256; ++b) total += histogram[uint2(b, row)];
        uint running = 0, median = 128;
        for (uint m = 0; m < 256; ++m) {
            running += histogram[uint2(m, row)];
            if (2 * running >= total) { median = m; break; }
        }
        // The foreground needs a body's worth of cells (about 1 % of a 320 x 180 grid).
        const uint needed = mode == 4 ? max(16u, gridW * gridH / 100u) : 1u;
        histogram[uint2(256, row)] = total >= needed ? median : (mode == 4 ? 1023u : 128u);
        return;
    }
    localBins[index] = 0;
    localBins[index + 256] = 0;
    GroupMemoryBarrierWithGroupSync();
    if (id.x < gridW && id.y < gridH) {
        const int2 v = int2(round(float2(rawFlow[id.xy]) * scale));
        bool counted = true;
        if (mode == 3) {
            // Middle of the picture (where a third-person character stands), unlike the camera.
            const float2 g = float2(int(histogram[uint2(256, 0)]) - 128, int(histogram[uint2(256, 1)]) - 128);
            const float2 at = (float2(id.xy) + .5) / float2(gridW, gridH);
            counted = all(at > float2(.2, .1)) && all(at < float2(.8, .95)) && dot(g, g) > 4 &&
                      length(float2(v) - g) > 2 + .1 * length(g);
        }
        if (counted) {
            const uint2 bin = uint2(clamp(v + 128, 0, 255));
            InterlockedAdd(localBins[bin.x], 1u);
            InterlockedAdd(localBins[256 + bin.y], 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint c = 0; c < 2; ++c) {
        const uint count = localBins[c * 256 + index];
        if (count != 0) InterlockedAdd(histogram[uint2(index, row0 + c)], count);
    }
}
