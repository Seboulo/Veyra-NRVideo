// SPDX-License-Identifier: GPL-3.0-only
// RIFE light output (RifeDmlBackend, VEYRA_TEST_RIFE_LIGHT): RIFE's own last step,
// at the working size, from the flow model's small output. On the picture at
// time t the model gives F_t->0 and F_t->1 (tensor pixels) and the mask (after
// its sigmoid: the weight of the previous frame). Both are upsampled bilinearly
// (clamped to the tensor content), the flows scaled to working-size pixels; the
// previous frame is read at x + F_t->0 and the current one at x + F_t->1
// (bilinear, edges repeated like grid_sample's border padding), and the result
// is mask * previous + (1 - mask) * current, as in Practical-RIFE's IFNet.

cbuffer Constants : register(b0)
{
    uint width;         // working size
    uint height;
    uint flowWidth;     // tensor content
    uint flowHeight;
    uint paddedWidth;   // model tensor
    uint paddedHeight;
    uint pad0;
    uint pad1;
};

Texture2D<float4> previous : register(t0);
Texture2D<float4> current : register(t1);
// The tensors stay in UNORDERED_ACCESS: DirectML writes them in that state.
RWByteAddressBuffer flow : register(u0);   // [1, 4, paddedHeight, paddedWidth] fp32
RWByteAddressBuffer mask : register(u1);   // [1, 1, paddedHeight, paddedWidth] fp32
RWTexture2D<float4> output : register(u2);

float4 flowAt(uint2 p)
{
    const uint plane = paddedWidth * paddedHeight;
    const uint index = p.y * paddedWidth + p.x;
    return float4(asfloat(flow.Load((0 * plane + index) * 4)), asfloat(flow.Load((1 * plane + index) * 4)),
                  asfloat(flow.Load((2 * plane + index) * 4)), asfloat(flow.Load((3 * plane + index) * 4)));
}

float maskAt(uint2 p) { return asfloat(mask.Load((p.y * paddedWidth + p.x) * 4)); }

float3 fetch(Texture2D<float4> frame, float2 p)
{
    const float2 last = float2(width - 1, height - 1);
    p = clamp(p, 0, last);
    const uint2 p0 = uint2(p);
    const uint2 p1 = min(p0 + 1, uint2(last));
    const float2 f = p - float2(p0);
    const float3 a = frame.Load(int3(p0, 0)).rgb, b = frame.Load(int3(p1.x, p0.y, 0)).rgb;
    const float3 c = frame.Load(int3(p0.x, p1.y, 0)).rgb, d = frame.Load(int3(p1, 0)).rgb;
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;
    const float2 scale = float2(flowWidth, flowHeight) / float2(width, height);
    const float2 last = float2(flowWidth - 1, flowHeight - 1);
    const float2 q = clamp((float2(id.xy) + 0.5) * scale - 0.5, 0, last);
    const uint2 q0 = uint2(q);
    const uint2 q1 = min(q0 + 1, uint2(last));
    const float2 f = q - float2(q0);
    const float4 F = lerp(lerp(flowAt(q0), flowAt(uint2(q1.x, q0.y)), f.x), lerp(flowAt(uint2(q0.x, q1.y)), flowAt(q1), f.x), f.y);
    const float M = saturate(lerp(lerp(maskAt(q0), maskAt(uint2(q1.x, q0.y)), f.x), lerp(maskAt(uint2(q0.x, q1.y)), maskAt(q1), f.x), f.y));
    const float3 a = fetch(previous, float2(id.xy) + F.xy / scale);
    const float3 b = fetch(current, float2(id.xy) + F.zw / scale);
    output[id.xy] = float4(saturate(a * M + b * (1.0 - M)), 1.0);
}
