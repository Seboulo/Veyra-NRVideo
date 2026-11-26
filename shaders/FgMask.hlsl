// Local fallback mask for frame generation (VEYRA_TEST_FG_MASK, needs VEYRA_TEST_FLOW_GLOBAL=1,
// FidelityFX vectors). Where the optical flow is unreliable, the generated frame shows the
// real frame instead (EffectExclusion writes it into XeSS's UI layer with this weight): the
// region moves at the source rate there, but no torn pattern is shown. This is the
// "artifact masking" of SVP ("more masking => more parts of interpolated frame is
// blended/replaced with source => less smoothness") and the pattern-level blend toward the
// repeated frame of television converters (US 8223831); no PC frame generation without game
// vectors repairs a lattice either (AMD AFMF 2 turns generation off for the whole frame).
// A grid cell is unreliable when its vector disagrees with the cells around it, in space or
// in its change since the last frame (a periodic texture makes the flow pick a different
// period in neighbouring cells and from frame to frame: OBS recordings 2026-11-26), and moves
// neither like the camera nor stays still (a third-person character's own cells and the
// background around it each agree with one of the two, so their boundary is not masked).
// The mask is grown by one cell at 60 % and upsampled bilinearly, which feathers its edge.
Texture2D<int2> rawFlow : register(t0);          // grid extent, FidelityFX integer pixels
Texture2D<int2> previousFlow : register(t1);     // grid extent, last frame's raw vectors
Texture2D<uint> globalMotion : register(t2);     // GlobalMotion.hlsl: x = 256 holds median + 128
RWTexture2D<float> mask : register(u0);          // grid extent, 0 = generated, 1 = real frame

cbuffer FgMaskConstants : register(b0)
{
    uint gridW;
    uint gridH;
    float threshold;     // pixels of disagreement where the weight starts (full at twice it)
    uint historyValid;   // 0: previousFlow is not last frame's (first frame, reset)
    uint unused0;
    uint unused1;
    uint unused2;
    uint unused3;
};

float median9(float v[9])
{
    [unroll] for (int i = 1; i < 9; ++i)
        [unroll] for (int j = i; j > 0; --j)
            if (v[j] < v[j - 1]) { const float t = v[j]; v[j] = v[j - 1]; v[j - 1] = t; }
    return v[4];
}

float unreliable(int2 cell, float2 camera)
{
    const int2 limit = int2(gridW - 1, gridH - 1);
    float vx[9], vy[9], cx[9], cy[9];
    [unroll] for (int k = 0; k < 9; ++k) {
        const int2 at = clamp(cell + int2(k % 3 - 1, k / 3 - 1), int2(0, 0), limit);
        const float2 v = float2(rawFlow[at]);
        const float2 change = historyValid != 0 ? v - float2(previousFlow[at]) : 0;
        vx[k] = v.x; vy[k] = v.y; cx[k] = change.x; cy[k] = change.y;
    }
    const float2 own = float2(vx[4], vy[4]);
    const float2 ownChange = float2(cx[4], cy[4]);
    // Cells that move like the camera or stay still belong to one of the two layers of a
    // third-person scene; the boundary between them is not an error.
    if (length(own - camera) <= threshold || length(own) <= 2) return 0;
    const float spatial = length(own - float2(median9(vx), median9(vy)));
    const float temporal = length(ownChange - float2(median9(cx), median9(cy)));
    return saturate((max(spatial, temporal) - threshold) / threshold);
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gridW || id.y >= gridH) return;
    const int2 cell = int2(id.xy);
    const float2 camera = float2(int(globalMotion[uint2(256, 0)]) - 128, int(globalMotion[uint2(256, 1)]) - 128);
    float weight = unreliable(cell, camera);
    if (weight < 1) {
        float around = 0;
        [unroll] for (int k = 0; k < 9; ++k)
            if (k != 4) around = max(around, unreliable(cell + int2(k % 3 - 1, k / 3 - 1), camera));
        weight = max(weight, .6 * around);
    }
    mask[cell] = weight;
}
