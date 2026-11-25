// Frame-generation motion refinement (XeSS, motion source "optical flow").
// Optical flow returns one current->previous vector per grid cell, so a cell that straddles
// a moving character and the background moves both with one vector. Per flow pixel, pick
// the vector of the 3x3 neighbouring cells whose 3x3 luma patch matches the previous
// frame best: motion boundaries follow the picture instead of the grid. Unlike
// NvofDensify nothing is zeroed; the best cost is kept for FlowFill to find pixels that
// have no correspondence at all (background uncovered by a moving object).
// With NVOF's backward output (previous->current) the chosen vector is also checked for
// forward-backward agreement: the previous-frame point it lands on must flow back to this
// pixel. Uncovered background has no vector that does - its true vector lands on the object
// that covered it, the object's vector on background that moved on - even where a wrong
// vector matches the smooth background beside it photometrically. The choice itself stays
// photometric: on rotating limbs and repetitive texture the check is too noisy to rank by.
// FidelityFX optical flow has no backward output; FlowClaim makes the same check for it.
// The zero vector is one more candidate (as in 3-D recursive search block matching, de Haan
// et al., IEEE TCSVT 1993): a block flow estimated coarse to fine gives a small object that
// stays still on screen (a third-person character or a HUD element while the camera turns)
// the background's motion in every neighbouring cell, so no cell candidate keeps it in place
// and frame generation moves it with the background. Zero wins only when its patch matches
// clearly better (zeroBias), so flat or repetitive background keeps its cell vector.
Texture2D<int2> rawFlow : register(t0);          // grid extent: NVOF S10.5 or FidelityFX integer pixels
Texture2D<float4> previousColor : register(t1);  // flow extent
Texture2D<float4> currentColor : register(t2);   // flow extent
Texture2D<int2> reverseFlow : register(t3);      // grid extent, S10.5, previous->current (NVOF)
Texture2D<uint> sceneChanges : register(t4);     // FidelityFX scene-change history at (1,0)
Texture2D<uint> globalMotion : register(t5);     // GlobalMotion.hlsl: x = 256 holds median + 128
RWTexture2D<float4> refined : register(u0);      // xy vector, z best cost, w forward-backward disagreement

cbuffer FlowRefineConstants : register(b0)
{
    uint gridW;
    uint gridH;
    uint flowW;
    uint flowH;
    uint gridSize;
    float tieBias;   // added to every neighbour cell's cost: the own cell wins ties
    uint flags;      // 1 reverseFlow holds NVOF's backward output (else w = 0); 2 FidelityFX
                     // vectors; 4 FidelityFX still warming up; 8 sub-pixel refinement;
                     // 16 small vectors around zero as well (needs zeroBias > 0);
                     // 32 camera (global) motion candidate and foreground preference
    float zeroBias;  // > 0: also try the zero vector, at this extra cost; 0: off
};

float luma(float3 c) { return dot(c, float3(.2126, .7152, .0722)); }

// Equiangular (V-shaped) fit of three matching costs at -1, 0, +1 pixel: the sub-pixel offset
// of the minimum. For an absolute-difference cost this line fit is less biased toward whole
// pixels than a parabola (Shimizu and Okutomi, "Sub-pixel estimation error cancellation on
// area-based matching", IJCV 2005). Returns 0 when the centre is not a clear minimum.
float subPixel(float minus, float centre, float plus)
{
    const float rise = max(minus, plus) - centre;
    if (rise <= 1e-4 || centre > min(minus, plus)) return 0;
    return clamp(.5 * (minus - plus) / rise, -.5, .5);
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= flowW || id.y >= flowH) return;
    const int2 p = int2(id.xy);
    const bool fidelity = (flags & 2u) != 0;
    // The frames NvofDensify zeroes for FidelityFX (warm-up and its four-frame cut history)
    // carry no motion here either.
    if ((flags & 4u) != 0 || (fidelity && (sceneChanges[uint2(1, 0)] & 15u) != 0)) { refined[p] = 0; return; }
    const float scale = fidelity ? 1.0 : 1.0 / 32.0;
    const int2 limit = int2(flowW - 1, flowH - 1);
    float current[9];
    [unroll] for (int j = 0; j < 9; ++j)
        current[j] = luma(currentColor[clamp(p + int2(j % 3 - 1, j / 3 - 1), int2(0, 0), limit)].rgb);
    const int2 cell = p / int(gridSize);
    const int2 cellLimit = int2(gridW - 1, gridH - 1);
    float bestCost = 1e9;
    float2 best = 0;
    [unroll] for (int k = 0; k < 9; ++k) {
        const int2 offset = int2(k % 3 - 1, k / 3 - 1);
        const float2 v = float2(rawFlow[clamp(cell + offset, int2(0, 0), cellLimit)]) * scale;
        const int2 q = p + int2(round(v));
        float cost = 0;
        [unroll] for (int j = 0; j < 9; ++j)
            cost += abs(current[j] - luma(previousColor[clamp(q + int2(j % 3 - 1, j / 3 - 1), int2(0, 0), limit)].rgb));
        cost /= 9;
        if (k != 4) cost += tieBias;
        if (cost < bestCost) { bestCost = cost; best = v; }
    }
    // Camera motion: one more candidate where the grid cells around are all wrong.
    const bool useGlobal = (flags & 32u) != 0;
    float2 global = 0;
    if (useGlobal) {
        global = float2(int(globalMotion[uint2(256, 0)]) - 128, int(globalMotion[uint2(256, 1)]) - 128);
        const int2 q = p + int2(global);
        float cost = 0;
        [unroll] for (int j = 0; j < 9; ++j)
            cost += abs(current[j] - luma(previousColor[clamp(q + int2(j % 3 - 1, j / 3 - 1), int2(0, 0), limit)].rgb));
        cost = cost / 9 + tieBias;
        if (cost < bestCost) { bestCost = cost; best = global; }
    }
    // Foreground preference (halo reduction in television frame-rate converters): when the
    // winner moves with the camera, a still or slow candidate that matches about as well
    // belongs to an object in front (a third-person character), not to the background.
    // Flat background that takes it moves invisibly. Otherwise zero must win clearly.
    const bool cameraWinner = useGlobal && dot(global, global) > 4 && length(best - global) <= 1.5;
    const float stillBias = cameraWinner ? -.002 : zeroBias;
    if (zeroBias > 0) {
        // The best still candidate (zero, then the near ones) is compared once with the
        // winner so far, so the bias applies once whatever its sign.
        float stillCost = 0;
        [unroll] for (int j = 0; j < 9; ++j)
            stillCost += abs(current[j] - luma(previousColor[clamp(p + int2(j % 3 - 1, j / 3 - 1), int2(0, 0), limit)].rgb));
        stillCost /= 9;
        float2 still = 0;
        // A character that stays near the screen centre while the camera turns still moves a
        // pixel or two (walk cycle, camera bob): the exact zero vector misses it and the
        // background's vector wins. The "update" candidates of 3-D recursive search (de Haan
        // et al. 1993) are small vectors around a candidate; here the 12 nearest to zero.
        if ((flags & 16u) != 0) {
            static const int2 kNear[12] = { int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1),
                int2(1, 1), int2(-1, 1), int2(1, -1), int2(-1, -1),
                int2(2, 0), int2(-2, 0), int2(0, 2), int2(0, -2) };
            [unroll] for (int n = 0; n < 12; ++n) {
                float nearCost = 0;
                [unroll] for (int j = 0; j < 9; ++j)
                    nearCost += abs(current[j] - luma(previousColor[clamp(p + kNear[n] + int2(j % 3 - 1, j / 3 - 1), int2(0, 0), limit)].rgb));
                nearCost /= 9;
                if (nearCost < stillCost) { stillCost = nearCost; still = float2(kNear[n]); }
            }
        }
        // Biased to choose; the unbiased cost goes on to FlowFill's no-correspondence test.
        if (stillCost + stillBias < bestCost) { bestCost = stillCost; best = still; }
    }
    // FidelityFX vectors are whole pixels. Interpolating halfway, a whole-pixel error
    // becomes a half-pixel misalignment of the two frames frame generation blends, which
    // softens every moving texture in the generated frames. Refine the chosen vector to a
    // fraction of a pixel from the costs one pixel either side. The zero vector stays exact:
    // still HUD and characters must not shimmer.
    if ((flags & 8u) != 0 && any(best != 0)) {
        const int2 q = p + int2(round(best));
        float c[4];
        [unroll] for (int s = 0; s < 4; ++s) {
            const int2 dir = s == 0 ? int2(-1, 0) : s == 1 ? int2(1, 0) : s == 2 ? int2(0, -1) : int2(0, 1);
            float cost = 0;
            [unroll] for (int j = 0; j < 9; ++j)
                cost += abs(current[j] - luma(previousColor[clamp(q + dir + int2(j % 3 - 1, j / 3 - 1), int2(0, 0), limit)].rgb));
            c[s] = cost / 9;
        }
        // bestCost of a neighbour cell carries tieBias; compare against the raw cost.
        float centre = 0;
        [unroll] for (int j = 0; j < 9; ++j)
            centre += abs(current[j] - luma(previousColor[clamp(q + int2(j % 3 - 1, j / 3 - 1), int2(0, 0), limit)].rgb));
        centre /= 9;
        best += float2(subPixel(c[0], centre, c[1]), subPixel(c[2], centre, c[3]));
    }
    float disagreement = 0;
    if ((flags & 1u) != 0) {
        const int2 back = clamp(int2(floor((float2(p) + .5 + best) / float(gridSize))), int2(0, 0), cellLimit);
        disagreement = length(best + float2(reverseFlow[back]) / 32.0);
    }
    refined[p] = float4(best, bestCost, disagreement);
}
