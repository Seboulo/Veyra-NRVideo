// Pseudo-depth for XeSS frame generation from captured video (VEYRA_TEST_FG_DEPTH=1).
// XeSS-FG upsamples low-res motion vectors and dilates them with the depth buffer: it keeps
// the motion of the foremost surface around each pixel (XeSS-FG developer guide, "Motion
// Vectors"). A capture has no depth, and a constant depth leaves that choice arbitrary, so the
// background's vector can win on the edge of a character that stays still while the camera
// turns. Picture-rate converters without depth separate a "background vector" from a
// "foreground vector" the same way (halo reduction, e.g. Philips US7010039): the motion that
// most of the surroundings share is background; a region moving differently is in front.
// Here each pixel compares its vector with 16 samples on two wide rings (about 100 and 220
// pixels): wider than a character's head or body, so they mostly land on the background. A
// pixel that agrees with most of them is background (far); one that does not is in front. The
// rings are wide on purpose: at an edge a narrow ring sees both sides about equally from either
// side and cannot tell which one is in front. A uniform pan agrees everywhere and keeps the
// constant far plane. Smaller depth is nearer (XeSS default).
// With the camera-motion estimate (VEYRA_TEST_FLOW_GLOBAL=1) a pixel that moves like the
// camera while it turns is background outright, and one that does not is in front; the
// rings then only grade how far in front. This is the foreground/background split of
// television frame-rate converters (global-motion candidate, Biezen and de Haan, IEEE TCSVT
// 1998), and it holds for a character larger than the rings, whose own pixels the rings see.
// With VEYRA_TEST_FG_DEPTH_MATCH=1 a pixel whose vector does not land on matching colour in the
// previous frame (3x3 patch, mean absolute RGB difference above matchLimit) is put on the far
// plane. Those are the background just uncovered behind a moving character, where the flow can
// only guess: they have no reliable motion, so they must not compete with the character for the
// foremost vector when XeSS dilates the motion at the character's edge (occlusion handling of
// picture-rate converters: uncovered areas take the background, de Haan, "Video processing for
// multimedia systems", 2000). Their vectors are left unchanged.
Texture2D<float2> motion : register(t0);   // work extent, current->previous pixels (XeSS input)
Texture2D<uint> globalMotion : register(t1); // GlobalMotion.hlsl (x = 256: median + 128), or null
Texture2D<float4> currentColor : register(t2);  // flow resolution, this frame
Texture2D<float4> previousColor : register(t3); // flow resolution, previous frame
RWTexture2D<float> depth : register(u0);   // work extent

cbuffer FgPseudoDepthConstants : register(b0)
{
    uint width;
    uint height;
    float innerRadius;  // pixels
    float outerRadius;  // pixels
    float tolerance;    // vectors closer than this (pixels) agree
    float globalScale;  // work pixels per flow pixel; 0 = no camera-motion estimate
    float matchScale;   // flow pixels per work pixel; 0 = no colour-match check
    float matchLimit;   // mean absolute RGB difference (0..1) above which a vector has no match
};

float matchError(int2 q, int2 d, int2 limit)
{
    if (any(q + d < 0) || any(q + d > limit)) return 1; // comes from outside the picture
    float e = 0;
    [unroll] for (int y = -1; y <= 1; ++y)
        [unroll] for (int x = -1; x <= 1; ++x) {
            const int2 a = clamp(q + int2(x, y), 0, limit);
            const int2 b = clamp(a + d, 0, limit);
            e += dot(abs(currentColor[a].rgb - previousColor[b].rgb), float3(1, 1, 1)) / 3;
        }
    return e / 9;
}

static const float2 kRing[8] = {
    float2(1, 0), float2(.7071, .7071), float2(0, 1), float2(-.7071, .7071),
    float2(-1, 0), float2(-.7071, -.7071), float2(0, -1), float2(.7071, -.7071) };

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;
    const int2 p = int2(id.xy);
    const int2 limit = int2(width - 1, height - 1);
    const float2 v = motion[p];
    float agree = 0, total = 0;
    [unroll] for (int r = 0; r < 2; ++r) {
        const float radius = r == 0 ? innerRadius : outerRadius;
        [unroll] for (int k = 0; k < 8; ++k) {
            const int2 q = p + int2(round(kRing[k] * radius));
            if (any(q < 0) || any(q > limit)) continue;
            // Tolerance grows with speed: fast background is never perfectly uniform.
            const float2 w = motion[q];
            agree += length(w - v) <= tolerance + .1 * max(length(v), length(w)) ? 1 : 0;
            total += 1;
        }
    }
    // A lone sample cannot tell front from back: such pixels stay at the far plane.
    const float share = total > 0 ? agree / total : 1;
    float z = lerp(.25, 1.0, share);
    if (globalScale > 0) {
        const float2 g = float2(int(globalMotion[uint2(256, 0)]) - 128, int(globalMotion[uint2(256, 1)]) - 128) * globalScale;
        // Only while the camera turns: with a still camera everything still agrees anyway.
        if (dot(g, g) > 4) {
            const bool camera = length(v - g) <= tolerance + .1 * max(length(v), length(g));
            z = camera ? 1.0 : lerp(.25, .75, share);
        }
    }
    if (matchScale > 0 && z < 1) {
        uint fw, fh;
        currentColor.GetDimensions(fw, fh);
        const int2 q = min(int2(float2(p) * matchScale), int2(fw - 1, fh - 1));
        if (matchError(q, int2(round(v * matchScale)), int2(fw - 1, fh - 1)) > matchLimit) z = 1.0;
    }
    depth[p] = z;
}
