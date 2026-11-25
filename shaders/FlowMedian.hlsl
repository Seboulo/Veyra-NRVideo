// Edge-aware weighted median of the frame-generation motion (VEYRA_TEST_FLOW_MEDIAN=1), after
// FlowFill. Sun, Roth and Black ("Secrets of optical flow estimation and their principles",
// CVPR 2010; IJCV 2014) found median filtering of the flow field to be the single step that
// most improves accuracy, and its weighted, non-local form (weights from colour similarity
// and distance) to keep motion boundaries on image edges. Here it does two jobs:
// - FlowRefine decides per pixel on a 3x3 patch. On flat or noisy areas (hair, skin, sky,
//   capture noise) several candidates match about equally and the choice flips from pixel
//   to pixel: frame generation then warps neighbouring pixels apart ("jelly", torn edges).
// - A still character in front of a turning background keeps the zero vector where it is
//   textured; its flat parts can take the background's vector. The median gives them the
//   vector the rest of the same-coloured region chose.
// Each component is the weighted median of 25 samples (5x5, two pixels apart: a 9x9 area).
// The weight of a sample falls with its colour distance to this pixel, so a sample across
// an edge (other object) barely counts and thin structures keep their own motion.
Texture2D<float2> flow : register(t0);           // flow extent, current->previous (FlowFill)
Texture2D<float4> currentColor : register(t1);   // flow extent
RWTexture2D<float2> output : register(u0);       // flow extent

cbuffer FlowMedianConstants : register(b0)
{
    uint flowW;
    uint flowH;
    float colourSigma;   // RGB distance (0..1 per channel) at which a sample weighs e^-0.5
    float spatialSigma;  // pixels
    uint unused0;
    uint unused1;
    uint unused2;
    uint unused3;
};

#define TAPS 25

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= flowW || id.y >= flowH) return;
    const int2 p = int2(id.xy);
    const int2 limit = int2(flowW - 1, flowH - 1);
    const float3 centre = currentColor[p].rgb;
    const float colourK = .5 / (colourSigma * colourSigma);
    const float spatialK = .5 / (spatialSigma * spatialSigma);
    float2 v[TAPS];
    float w[TAPS];
    [unroll] for (int i = 0; i < TAPS; ++i) {
        const int2 offset = int2(i % 5 - 2, i / 5 - 2) * 2;
        const int2 q = clamp(p + offset, int2(0, 0), limit);
        v[i] = flow[q];
        const float3 d = currentColor[q].rgb - centre;
        w[i] = exp(-colourK * dot(d, d) - spatialK * float(dot(offset, offset)));
    }
    // Weighted median per component: the sample value with the smallest weighted sum of
    // absolute differences to all samples (the L1 minimiser among the samples).
    float2 best = v[12];
    float2 bestCost = 1e30;
    [unroll] for (int k = 0; k < TAPS; ++k) {
        float2 cost = 0;
        [unroll] for (int j = 0; j < TAPS; ++j) cost += w[j] * abs(v[k] - v[j]);
        if (cost.x < bestCost.x) { bestCost.x = cost.x; best.x = v[k].x; }
        if (cost.y < bestCost.y) { bestCost.y = cost.y; best.y = v[k].y; }
    }
    output[p] = best;
}
