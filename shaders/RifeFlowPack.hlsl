// SPDX-License-Identifier: GPL-3.0-only
// XeSS motion from RIFE's flow (RifeFlow): the previous and current optical-flow
// input frames (B8G8R8A8, encoded colour) -> the flow model's planar fp32 inputs
// img0 (previous), img1 (current) and the timestep plane. The tensor is smaller
// than the input: each tensor pixel averages its input footprint with n x n
// Load taps (n = ceil(footprint), at most 8; an exact box average for whole
// ratios such as 2560 -> 640). Padding repeats the last row and column, as in
// RifePack.hlsl.

cbuffer Constants : register(b0)
{
    uint inputWidth;
    uint inputHeight;
    uint flowWidth;     // tensor content
    uint flowHeight;
    uint paddedWidth;   // model tensor
    uint paddedHeight;
    float timestep;
    uint pad0;
};

Texture2D<float4> previous : register(t0);
Texture2D<float4> current : register(t1);
RWByteAddressBuffer img0 : register(u0);
RWByteAddressBuffer img1 : register(u1);
RWByteAddressBuffer times : register(u2);

float3 area(Texture2D<float4> frame, uint2 at)
{
    const float2 footprint = float2(inputWidth, inputHeight) / float2(flowWidth, flowHeight);
    const uint2 taps = clamp(uint2(ceil(footprint - 1e-3)), 1u, 8u);
    const uint2 last = uint2(inputWidth - 1, inputHeight - 1);
    float3 sum = 0;
    for (uint j = 0; j < taps.y; ++j)
        for (uint i = 0; i < taps.x; ++i) {
            const float2 p = float2(at) * footprint + (float2(i, j) + 0.5) * footprint / float2(taps);
            sum += saturate(frame.Load(int3(min(uint2(p), last), 0)).rgb);
        }
    return sum / float(taps.x * taps.y);
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= paddedWidth || id.y >= paddedHeight) return;
    const uint plane = paddedWidth * paddedHeight;
    const uint index = id.y * paddedWidth + id.x;
    const uint2 at = min(id.xy, uint2(flowWidth - 1, flowHeight - 1));
    const float3 a = area(previous, at);
    const float3 b = area(current, at);
    [unroll] for (uint c = 0; c < 3; ++c) {
        img0.Store((c * plane + index) * 4, asuint(a[c]));
        img1.Store((c * plane + index) * 4, asuint(b[c]));
    }
    times.Store(index * 4, asuint(timestep));
}
