// SPDX-License-Identifier: GPL-3.0-only
// RIFE light (RifeDmlBackend, VEYRA_TEST_RIFE_LIGHT): the enhanced colour frame ->
// the flow model's planar fp32 RGB input at 1/divisor of the working size. Each
// tensor pixel averages its input footprint with n x n Load taps (n =
// ceil(footprint), at most 8: an exact box average for whole ratios such as
// 2560 -> 640), as RifeFlowPack.hlsl. Padding repeats the last row and column,
// as RifePack.hlsl.

cbuffer Constants : register(b0)
{
    uint inputWidth;    // colour frame
    uint inputHeight;
    uint flowWidth;     // tensor content
    uint flowHeight;
    uint paddedWidth;   // model tensor
    uint paddedHeight;
    uint pad0;
    uint pad1;
};

Texture2D<float4> color : register(t0);
RWByteAddressBuffer planes : register(u0);

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= paddedWidth || id.y >= paddedHeight) return;
    const uint plane = paddedWidth * paddedHeight;
    const uint index = id.y * paddedWidth + id.x;
    const uint2 at = min(id.xy, uint2(flowWidth - 1, flowHeight - 1));
    const float2 footprint = float2(inputWidth, inputHeight) / float2(flowWidth, flowHeight);
    const uint2 taps = clamp(uint2(ceil(footprint - 1e-3)), 1u, 8u);
    const uint2 last = uint2(inputWidth - 1, inputHeight - 1);
    float3 sum = 0;
    for (uint j = 0; j < taps.y; ++j)
        for (uint i = 0; i < taps.x; ++i) {
            const float2 p = float2(at) * footprint + (float2(i, j) + 0.5) * footprint / float2(taps);
            sum += saturate(color.Load(int3(min(uint2(p), last), 0)).rgb);
        }
    const float3 rgb = sum / float(taps.x * taps.y);
    planes.Store((0 * plane + index) * 4, asuint(rgb.r));
    planes.Store((1 * plane + index) * 4, asuint(rgb.g));
    planes.Store((2 * plane + index) * 4, asuint(rgb.b));
}
