// Test view of the frame-generation motion (VEYRA_TEST_FLOW_VIEW=1): blends the
// current->previous vector of every work pixel over the picture, coloured as in the
// Middlebury optical-flow benchmark (Baker et al., IJCV 2011): hue = direction,
// saturation = length. Still pixels keep their own colours, so a third-person character
// that the flow leaves in place stays uncoloured while the turning background is tinted.
// Frame generation interpolates the tinted frames too; every shown frame carries the view.
Texture2D<float2> motion : register(t0);      // work extent, current->previous pixels
RWTexture2D<float4> frame : register(u0);     // work extent, SDR or HDR output colour

cbuffer FlowViewConstants : register(b0)
{
    uint width;
    uint height;
    float fullLength;  // vector length (pixels) shown at full saturation
    float opacity;     // tint strength at full saturation
    uint unused0;
    uint unused1;
    uint unused2;
    uint unused3;
};

float3 hue(float h)
{
    const float3 k = saturate(abs(frac(h + float3(0, 2.0 / 3.0, 1.0 / 3.0)) * 6 - 3) - 1);
    return k;
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;
    const float2 v = motion[id.xy];
    const float len = length(v);
    if (len < .25) return;
    const float amount = saturate(len / fullLength) * opacity;
    const float h = atan2(-v.y, -v.x) / 6.28318530718 + .5;
    const float4 colour = frame[id.xy];
    const float l = max(dot(colour.rgb, float3(.2126, .7152, .0722)), .15);
    frame[id.xy] = float4(lerp(colour.rgb, hue(h) * (.35 + l), amount), colour.a);
}
