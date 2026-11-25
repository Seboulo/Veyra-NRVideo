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
Texture2D<float2> motion : register(t0);   // work extent, current->previous pixels (XeSS input)
RWTexture2D<float> depth : register(u0);   // work extent

cbuffer FgPseudoDepthConstants : register(b0)
{
    uint width;
    uint height;
    float innerRadius;  // pixels
    float outerRadius;  // pixels
    float tolerance;    // vectors closer than this (pixels) agree
    uint unused0;
    uint unused1;
    uint unused2;
};

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
    depth[p] = lerp(.25, 1.0, share);
}
