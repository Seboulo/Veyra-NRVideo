// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <d3d12.h>
#include <cstdint>
#include <memory>
#include <string>

namespace veyra::gfx {
// RIFE video frame interpolation (Practical-RIFE, MIT) run by ONNX Runtime with
// the DirectML execution provider on the graph's own D3D12 queue. The runtime
// (onnxruntime.dll, DirectML.dll) and the fp16 models live in runtime\rife next
// to the application and are loaded by absolute path; nothing is linked.
//
// Per real frame: capture() records a copy of the colour frame and its
// conversion into the model's planar input, the owner submits that list, then
// for each subframe generate() queues the network on the same queue (no CPU
// wait: ONNX Runtime only records and submits) and copyOutput() records the
// conversion of its result into the owner's texture. Queue order alone orders
// the three steps. 2X and 3X: subframe k of n is the frame at time k/n.
//
// RIFE light (VEYRA_TEST_RIFE_LIGHT=d, 2..8): the same steps with the flow-only
// model (<model>_flow_fp16.onnx) on area-averaged frames at 1/d of the working
// size, then RifeLightWarp.hlsl does the network's own last step at the full
// size: both frames warped by the upsampled flows and blended with the mask.
// The network's cost falls with d squared (1/16 at d = 4). Offline on GTA
// captures (2026-11-27): d = 2 matched the full network on a lattice and beat it
// on fast turns around the character; d = 4 lost 0.5 dB on the lattice.
class RifeDmlBackend {
public:
    RifeDmlBackend();
    ~RifeDmlBackend();
    RifeDmlBackend(const RifeDmlBackend&) = delete;
    RifeDmlBackend& operator=(const RifeDmlBackend&) = delete;
    // runtime\rife holds the runtime DLLs and the model (file checks only).
    static bool runtimeInstalled();
    // SDR colour only (R8G8B8A8_UNORM); multiplier 2 or 3.
    bool initialize(ID3D12Device*, ID3D12CommandQueue*, uint32_t width, uint32_t height, DXGI_FORMAT, unsigned multiplier);
    // `color` is in COPY_SOURCE; the list runs on the queue given to initialize().
    bool capture(ID3D12GraphicsCommandList*, ID3D12Resource* color);
    // After the list holding capture() was submitted: queues subframe `sub` of
    // `of` (time sub/of) between the previous and this frame. `of` is the
    // multiplier, or 2 for a pair the owner reduced to 2X. False on failure.
    bool generate(unsigned sub, unsigned of);
    // Records the conversion of that subframe into `output` (UNORDERED_ACCESS,
    // same format as initialize()).
    bool copyOutput(ID3D12GraphicsCommandList*, ID3D12Resource* output, unsigned sub, unsigned of);
    // Drops the previous frame: the next capture starts a new pair.
    void reset();
    // A previous frame exists, so generate() has a real pair.
    bool hasPair() const;
    bool ready() const;
    const std::string& modelName() const;
    void shutdown();
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// XeSS frame generation motion from RIFE's flow (VEYRA_TEST_FLOW_RIFE=1). The
// flow-only model (tools/rife_models/make_rife_flow_models.py, default
// rife_v4.26_flow_fp16.onnx, VEYRA_TEST_FLOW_RIFE_MODEL overrides) runs on the
// previous and current optical-flow input frames at 1/divisor of the motion
// size with t = 0.5. RifeFlowToMotion.hlsl turns its two flows (from the middle
// picture to each frame) into current->previous motion in motion-texture
// pixels: per pixel x, mv = v(x + mv/2) with v = F_t->0 - F_t->1, solved by
// fixed-point sampling. Dense per-pixel vectors where the FidelityFX grid gives
// one integer vector per 8x8 cell (repetitive textures, fast moving edges).
// Per real frame: pack() records the model inputs, the owner submits, run()
// queues the network on the same queue (no CPU wait), convert() records the
// motion. Queue order alone orders the three steps, as for RifeDmlBackend.
class RifeFlow {
public:
    RifeFlow();
    ~RifeFlow();
    RifeFlow(const RifeFlow&) = delete;
    RifeFlow& operator=(const RifeFlow&) = delete;
    // previous/current: fixed B8G8R8A8_UNORM textures of equal size; motion: a
    // fixed R16G16_FLOAT texture. divisor 2..8 (4: a quarter of the motion size).
    bool initialize(ID3D12Device*, ID3D12CommandQueue*, ID3D12Resource* previous, ID3D12Resource* current,
                    ID3D12Resource* motion, unsigned divisor);
    // previous and current in NON_PIXEL_SHADER_RESOURCE.
    bool pack(ID3D12GraphicsCommandList*);
    // After the list holding pack() was submitted.
    bool run();
    // motion in UNORDERED_ACCESS.
    bool convert(ID3D12GraphicsCommandList*);
    bool ready() const;
    const std::string& modelName() const;
    void shutdown();
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}
