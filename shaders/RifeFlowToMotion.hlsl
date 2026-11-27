// SPDX-License-Identifier: GPL-3.0-only
// XeSS motion from RIFE's flow (RifeFlow): the flow-only model's output at
// t = 0.5, F_t->0 (x, y) and F_t->1 (x, y) in tensor pixels on the middle
// picture's grid, -> current->previous motion in motion-texture pixels.
// v = F_t->0 - F_t->1 is the previous-minus-current displacement of the content
// at a middle-grid point; a current pixel x sits at x + mv/2 on that grid, so
// mv(x) = v(x + mv/2), solved by fixed-point iteration from mv = v(x) with
// bilinear sampling (clamped to the tensor content). Offline on Sebastien's
// captures (2026-11-27): three iterations, quarter size, fp16 model.

cbuffer Constants : register(b0)
{
    uint motionWidth;
    uint motionHeight;
    uint flowWidth;     // tensor content
    uint flowHeight;
    uint paddedWidth;   // model tensor
    uint paddedHeight;
    uint iterations;
    uint pad0;
};

// The tensor stays in UNORDERED_ACCESS: DirectML writes it in that state.
RWByteAddressBuffer flow : register(u0);   // [1, 4, paddedHeight, paddedWidth] fp32
RWTexture2D<float2> motion : register(u1);

float2 displacement(uint2 p)
{
    const uint plane = paddedWidth * paddedHeight;
    const uint index = p.y * paddedWidth + p.x;
    const float2 toPrevious = float2(asfloat(flow.Load((0 * plane + index) * 4)), asfloat(flow.Load((1 * plane + index) * 4)));
    const float2 toCurrent = float2(asfloat(flow.Load((2 * plane + index) * 4)), asfloat(flow.Load((3 * plane + index) * 4)));
    return toPrevious - toCurrent;
}

float2 sampleDisplacement(float2 p)
{
    const float2 last = float2(flowWidth - 1, flowHeight - 1);
    p = clamp(p, 0, last);
    const uint2 p0 = uint2(p);
    const uint2 p1 = min(p0 + 1, uint2(last));
    const float2 f = p - float2(p0);
    return lerp(lerp(displacement(p0), displacement(uint2(p1.x, p0.y)), f.x),
                lerp(displacement(uint2(p0.x, p1.y)), displacement(p1), f.x), f.y);
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= motionWidth || id.y >= motionHeight) return;
    const float2 scale = float2(flowWidth, flowHeight) / float2(motionWidth, motionHeight);
    const float2 p = (float2(id.xy) + 0.5) * scale - 0.5;
    float2 mv = sampleDisplacement(p);
    for (uint i = 0; i < iterations; ++i) mv = sampleDisplacement(p + mv * 0.5);
    motion[id.xy] = mv / scale;
}
