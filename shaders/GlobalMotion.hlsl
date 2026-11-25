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
    uint mode;     // 0 clear, 1 accumulate, 2 resolve
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
        if (id.x <= 256 && id.y < 2) histogram[id.xy] = id.x == 256 ? 128u : 0u;
        return;
    }
    if (mode == 2) {
        // One thread per component walks the cumulative count to the middle sample.
        if (id.x != 0 || id.y >= 2) return;
        uint total = 0;
        for (uint b = 0; b < 256; ++b) total += histogram[uint2(b, id.y)];
        uint running = 0, median = 128;
        for (uint m = 0; m < 256; ++m) {
            running += histogram[uint2(m, id.y)];
            if (2 * running >= total) { median = m; break; }
        }
        histogram[uint2(256, id.y)] = total > 0 ? median : 128u;
        return;
    }
    localBins[index] = 0;
    localBins[index + 256] = 0;
    GroupMemoryBarrierWithGroupSync();
    if (id.x < gridW && id.y < gridH) {
        const int2 v = int2(round(float2(rawFlow[id.xy]) * scale));
        const uint2 bin = uint2(clamp(v + 128, 0, 255));
        InterlockedAdd(localBins[bin.x], 1u);
        InterlockedAdd(localBins[256 + bin.y], 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint c = 0; c < 2; ++c) {
        const uint count = localBins[c * 256 + index];
        if (count != 0) InterlockedAdd(histogram[uint2(index, c)], count);
    }
}
