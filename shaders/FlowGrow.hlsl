// Foreground growth of the FG motion (VEYRA_TEST_FLOW_GROW=1, needs VEYRA_TEST_FLOW_GLOBAL=1).
// While the camera orbits a third-person character, the optical flow's 8x8 cells next to the
// character carry the background's vector, and on smooth skin or motion-blurred edges that
// vector matches the previous frame about as well as the character's own: FlowRefine then
// keeps it. Frame generation moves those pixels with the background and the face turns
// transparent on the side that faces the motion (OBS recording 2026-11-26).
// Here a pixel that moves like the camera tries the vectors of nearby pixels that do not
// (32 samples on four rings up to about 40 pixels): the spatial-propagation step of
// PatchMatch (Barnes et al., SIGGRAPH 2009) and the spatial candidates of 3-D recursive
// search (de Haan et al., IEEE TCSVT 1993), restricted to foreground vectors. The best of them
// replaces the camera vector when its 3x3 luma patch matches about as well (growBias: the
// foreground preference of television frame-rate converters, halo reduction). Textured
// background keeps the camera vector because the character's vector does not match it; smooth
// background may follow the character over a few pixels, where it cannot be seen.
// Depth-based motion dilation does the same in game frame generation (FSR 3 and XeSS dilate
// vectors toward the nearest surface); a capture has no depth, the colour match decides.
Texture2D<float4> refined : register(t0);        // FlowRefine: xy vector, z best cost, w disagreement
Texture2D<float4> previousColor : register(t1);  // flow extent
Texture2D<float4> currentColor : register(t2);   // flow extent
Texture2D<uint> globalMotion : register(t3);     // GlobalMotion.hlsl: x = 256 holds median + 128
RWTexture2D<float4> grown : register(u0);

cbuffer FlowGrowConstants : register(b0)
{
    uint flowW;
    uint flowH;
    float growBias;     // a foreground vector may match this much worse (mean luma 0..1) and win
    float radiusScale;  // ring radii are 4, 10, 20, 36 pixels times this
    uint unused0;
    uint unused1;
    uint unused2;
    uint unused3;
};

float luma(float3 c) { return dot(c, float3(.2126, .7152, .0722)); }

static const float2 kDirections[8] = {
    float2(1, 0), float2(.7071, .7071), float2(0, 1), float2(-.7071, .7071),
    float2(-1, 0), float2(-.7071, -.7071), float2(0, -1), float2(.7071, -.7071) };
static const float kRadii[4] = { 4, 10, 20, 36 };

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= flowW || id.y >= flowH) return;
    const int2 p = int2(id.xy);
    const float4 own = refined[p];
    const float2 g = float2(int(globalMotion[uint2(256, 0)]) - 128, int(globalMotion[uint2(256, 1)]) - 128);
    const float tolerance = 1.5 + .1 * length(g);
    // Only while the camera turns, and only pixels that move with it.
    if (dot(g, g) <= 4 || length(own.xy - g) > tolerance) { grown[p] = own; return; }
    const int2 limit = int2(flowW - 1, flowH - 1);
    float current[9];
    [unroll] for (int j = 0; j < 9; ++j)
        current[j] = luma(currentColor[clamp(p + int2(j % 3 - 1, j / 3 - 1), int2(0, 0), limit)].rgb);
    float ownCost = 0;
    {
        const int2 q = p + int2(round(own.xy));
        [unroll] for (int j = 0; j < 9; ++j)
            ownCost += abs(current[j] - luma(previousColor[clamp(q + int2(j % 3 - 1, j / 3 - 1), int2(0, 0), limit)].rgb));
        ownCost /= 9;
    }
    float bestCost = 1e9;
    float2 best = own.xy;
    float2 tried = g;   // the last vector tested: neighbouring samples often repeat it
    [loop] for (int r = 0; r < 4; ++r) {
        [unroll] for (int k = 0; k < 8; ++k) {
            const int2 at = clamp(p + int2(round(kDirections[k] * kRadii[r] * radiusScale)), int2(0, 0), limit);
            const float2 u = refined[at].xy;
            if (length(u - g) <= tolerance || length(u - tried) < .5) continue;
            tried = u;
            const int2 q = p + int2(round(u));
            float cost = 0;
            [unroll] for (int j = 0; j < 9; ++j)
                cost += abs(current[j] - luma(previousColor[clamp(q + int2(j % 3 - 1, j / 3 - 1), int2(0, 0), limit)].rgb));
            cost /= 9;
            if (cost < bestCost) { bestCost = cost; best = u; }
        }
    }
    grown[p] = bestCost <= ownCost + growBias ? float4(best, bestCost, own.w) : own;
}
