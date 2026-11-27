// SPDX-License-Identifier: GPL-3.0-only
#include "veyra/gfx/RifeDmlBackend.h"

#include "veyra/Log.h"
#include "veyra/RuntimePaths.h"
#include "veyra/pipeline/GpuPassUtils.h"

#include <windows.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef VEYRA_HAS_RIFE_DML
#include <DirectML.h>
#include <onnxruntime_c_api.h>
#include <dml_provider_factory.h>
#endif

namespace veyra::gfx {
using Microsoft::WRL::ComPtr;

namespace {
std::filesystem::path runtimeDirectory() { return runtime::applicationRoot() / "runtime" / "rife"; }
// 2X takes the full RIFE 4.26; 3X the lite 4.22, which leaves more of the
// 33 ms of a 30 fps pair for the rest of the graph (RX 9070 XT, 2026-11-22:
// 11.4 vs 8.9 ms per frame at 1088x1920 fp16).
std::string defaultModel(unsigned multiplier) { return multiplier > 2 ? "rife_v4.22_lite" : "rife_v4.26"; }
std::string selectedModel(unsigned multiplier) {
    wchar_t value[64]{};
    if (GetEnvironmentVariableW(L"VEYRA_TEST_RIFE_MODEL", value, 64)) {
        std::string name;
        for (const wchar_t* c = value; *c; ++c) name.push_back(char(*c));
        return name;
    }
    return defaultModel(multiplier);
}
// Model tensors are padded to whole blocks of the coarsest flow level: each IFBlock
// resizes by 1/scale and then halves twice, so the multiple is 4x its largest scale.
// 4.26 and 4.22 lite (largest scale 16): 64. 4.25 lite (32) and the half-resolution
// flow variants made by tools/rife_models/make_rife_scale_models.py ("_s05": every
// scale doubled, like Practical-RIFE --scale 0.5): 128, and 256 for 4.25 lite _s05.
// VEYRA_TEST_RIFE_PAD overrides it for other test models.
uint32_t padMultiple(const std::string& model) {
    wchar_t value[16]{};
    if (GetEnvironmentVariableW(L"VEYRA_TEST_RIFE_PAD", value, 16)) {
        const unsigned v = unsigned(_wtoi(value));
        if (v >= 64 && v <= 512 && (v & (v - 1)) == 0) return v;
    }
    const bool lite25 = model.find("4.25_lite") != std::string::npos;
    const bool half = model.find("_s05") != std::string::npos;
    return 64u << ((lite25 ? 1 : 0) + (half ? 1 : 0));
}
uint32_t padded(uint32_t v, uint32_t multiple) { return (v + multiple - 1) / multiple * multiple; }
}

#ifdef VEYRA_HAS_RIFE_DML
namespace {
// RIFE light (VEYRA_TEST_RIFE_LIGHT=2..8): the flow-only model (<model>_flow_fp16.onnx,
// tools/rife_models/make_rife_flow_models.py) runs at 1/divisor of the working size and
// RifeLightWarp.hlsl does RIFE's own last step (warp both frames, blend with the mask)
// at the full size. 0: the full network.
unsigned lightDivisor() {
    wchar_t value[8]{};
    if (!GetEnvironmentVariableW(L"VEYRA_TEST_RIFE_LIGHT", value, 8)) return 0;
    const unsigned v = unsigned(_wtoi(value));
    return v >= 2 && v <= 8 ? v : 0;
}
struct Runtime {
    const OrtApi* api = nullptr;
    const OrtDmlApi* dml = nullptr;
    OrtEnv* env = nullptr;
    decltype(&DMLCreateDevice) createDevice = nullptr;
    std::string error;
};
const Runtime& runtime() {
    static Runtime r;
    static std::once_flag once;
    std::call_once(once, [] {
        const auto dir = runtimeDirectory();
        // DirectML first, from runtime\rife by absolute path: onnxruntime.dll then
        // binds the already loaded module, never the older System32 DirectML.dll.
        HMODULE directMl = LoadLibraryExW((dir / L"DirectML.dll").c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        HMODULE ort = directMl ? LoadLibraryExW((dir / L"onnxruntime.dll").c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32) : nullptr;
        if (!directMl || !ort) { r.error = std::format("runtime\\rife DLLs not loaded err={}", GetLastError()); return; }
        r.createDevice = reinterpret_cast<decltype(&DMLCreateDevice)>(GetProcAddress(directMl, "DMLCreateDevice"));
        auto base = reinterpret_cast<const OrtApiBase*(ORT_API_CALL*)()>(GetProcAddress(ort, "OrtGetApiBase"));
        r.api = base ? base()->GetApi(ORT_API_VERSION) : nullptr;
        if (!r.createDevice || !r.api) { r.error = "DMLCreateDevice / OrtGetApiBase missing or ONNX Runtime too old"; r.api = nullptr; return; }
        if (auto* s = r.api->GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&r.dml))) {
            r.error = r.api->GetErrorMessage(s); r.api->ReleaseStatus(s); r.api = nullptr; return;
        }
        if (auto* s = r.api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "veyra-rife", &r.env)) {
            r.error = r.api->GetErrorMessage(s); r.api->ReleaseStatus(s); r.api = nullptr; return;
        }
        log::info("rife", std::format("runtime loaded dir={} ort={}", dir.string(), base()->GetVersionString()));
    });
    return r;
}
}

struct RifeDmlBackend::Impl {
    const Runtime* rt = nullptr;
    ID3D12Device* device = nullptr;
    ComPtr<IDMLDevice> dmlDevice;
    OrtSessionOptions* options = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* memory = nullptr;
    uint32_t width = 0, height = 0, pw = 0, ph = 0;
    // Tensor content: the working size, or 1/light of it in light mode.
    uint32_t flowW = 0, flowH = 0;
    unsigned light = 0;        // VEYRA_TEST_RIFE_LIGHT divisor, 0: the full network
    unsigned multiplier = 2;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    std::string model;
    // planes[2]: the two most recent frames. times/outputs per timestep: k/multiplier
    // for k = 1..multiplier-1, then 1/2 at 3X for pairs the owner reduces to 2X.
    struct Tensor { ComPtr<ID3D12Resource> buffer; void* allocation = nullptr; OrtValue* value = nullptr; };
    std::array<Tensor, 2> planes;
    // outputs: the frame (3 channels), or in light mode the flow (4: F_t->0, F_t->1)
    // with masks (1: the blend weight of the previous frame).
    std::vector<Tensor> times, outputs, masks;
    // bindings[current * steps + step]: img0 = planes[1-current], img1 = planes[current].
    std::vector<OrtIoBinding*> bindings;
    ComPtr<ID3D12Resource> input;   // copy of the colour frame (SRV for the pack pass)
    D3D12_RESOURCE_STATES inputState = D3D12_RESOURCE_STATE_COPY_DEST;
    pipeline::ComputePass pack, unpack, shot;
    // Light mode: frames[2] hold the two most recent colour frames at the working size
    // (frames[current] the newest), replacing input. lightPack heap: [0..1] frames SRV,
    // [2..3] planes UAV. warp heap: [2c, 2c+1] previous and current SRV when frames[c]
    // is the newest, then per (owner texture, subframe): flow, mask and texture UAVs.
    std::array<ComPtr<ID3D12Resource>, 2> frames;
    std::array<D3D12_RESOURCE_STATES, 2> frameStates{D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COMMON};
    pipeline::ComputePass lightPack, warp;
    ComPtr<ID3D12DescriptorHeap> staging;   // light-mode SRVs, created here and copied (see DescriptorStager)
    // Shot-change test (RifeShotChange.hlsl): work holds two 128x72 luma pictures and
    // 32 block errors; cut holds the 64-bit flag of the last pair at offset 0.
    ComPtr<ID3D12Resource> work, cut;
    D3D12_RESOURCE_STATES cutState = D3D12_RESOURCE_STATE_COMMON;
    std::unordered_map<ID3D12Resource*, UINT> outputSlots;   // unpack UAV per owner texture
    unsigned current = 1;      // planes index written by the last capture
    bool haveFrame = false;    // a capture since the last reset
    bool havePrevious = false; // ... and one before it: planes hold a real pair
    bool timesFilled = false, ok = false;

    unsigned steps() const { return multiplier > 2 ? multiplier : 1u; }
    int step(unsigned sub, unsigned of) const {
        if (sub < 1 || sub >= of) return -1;
        if (of == multiplier) return int(sub - 1);
        if (of == 2 && multiplier > 2) return int(multiplier - 1);
        return -1;
    }
    float time(unsigned step) const { return step + 1 < multiplier ? float(step + 1) / float(multiplier) : 0.5f; }
    bool check(OrtStatus* s, const char* what) {
        if (!s) return true;
        log::error("rife", std::format("{} failed: {}", what, rt->api->GetErrorMessage(s)));
        rt->api->ReleaseStatus(s);
        return false;
    }
    bool makeTensor(Tensor& t, int64_t channels) {
        const uint64_t bytes = uint64_t(channels) * pw * ph * 4;
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rd{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
        if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&t.buffer)))) {
            log::error("rife", std::format("tensor buffer {} bytes failed", bytes));
            return false;
        }
        if (!check(rt->dml->CreateGPUAllocationFromD3DResource(t.buffer.Get(), &t.allocation), "CreateGPUAllocationFromD3DResource")) return false;
        const int64_t shape[4] = {1, channels, int64_t(ph), int64_t(pw)};
        return check(rt->api->CreateTensorWithDataAsOrtValue(memory, t.allocation, size_t(bytes), shape, 4,
                                                             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t.value), "CreateTensor");
    }
    void release(Tensor& t) {
        if (t.value) rt->api->ReleaseValue(t.value);
        if (t.allocation) rt->dml->FreeGPUAllocation(t.allocation);
        t = {};
    }
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(const pipeline::ComputePass& pass, UINT slot) const {
        return {pass.heap->GetGPUDescriptorHandleForHeapStart().ptr + uint64_t(slot) * pass.increment};
    }
    D3D12_CPU_DESCRIPTOR_HANDLE cpu(const pipeline::ComputePass& pass, UINT slot) const {
        return {pass.heap->GetCPUDescriptorHandleForHeapStart().ptr + uint64_t(slot) * pass.increment};
    }
    void rawUav(ID3D12Resource* buffer, uint64_t channels, D3D12_CPU_DESCRIPTOR_HANDLE handle) const {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
        u.Format = DXGI_FORMAT_R32_TYPELESS;
        u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        u.Buffer.NumElements = UINT(channels * pw * ph);
        u.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device->CreateUnorderedAccessView(buffer, nullptr, &u, handle);
    }
    void copyStagedSrv(UINT stagingSlot, const pipeline::ComputePass& pass, UINT slot) const {
        const UINT increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const D3D12_CPU_DESCRIPTOR_HANDLE from{staging->GetCPUDescriptorHandleForHeapStart().ptr + uint64_t(stagingSlot) * increment};
        device->CopyDescriptorsSimple(1, cpu(pass, slot), from, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
    static void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* r, D3D12_RESOURCE_STATES& state, D3D12_RESOURCE_STATES to) {
        if (state == to) return;
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = r; b.Transition.StateBefore = state; b.Transition.StateAfter = to;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        list->ResourceBarrier(1, &b);
        state = to;
    }
    void dispatch(ID3D12GraphicsCommandList* list, const pipeline::ComputePass& pass, const uint32_t (&constants)[8],
                  D3D12_GPU_DESCRIPTOR_HANDLE srv, D3D12_GPU_DESCRIPTOR_HANDLE uav, uint32_t w, uint32_t h) const {
        pass.bind(list, reinterpret_cast<const float*>(constants), srv.ptr, uav.ptr);
        list->Dispatch((w + 15) / 16, (h + 15) / 16, 1);
    }
};

// Pack heap: [0] input SRV, [1..2] planes UAV, [3] null SRV, [4..] time UAVs.
// Unpack heap: [0] null SRV, then per (sub, owner texture): output UAV, texture UAV.
RifeDmlBackend::RifeDmlBackend() : p_(std::make_unique<Impl>()) {}
RifeDmlBackend::~RifeDmlBackend() { shutdown(); }

bool RifeDmlBackend::runtimeInstalled() {
    const auto dir = runtimeDirectory();
    std::error_code ec;
    return std::filesystem::exists(dir / L"onnxruntime.dll", ec) && std::filesystem::exists(dir / L"DirectML.dll", ec) &&
           std::filesystem::exists(dir / (defaultModel(2) + "_fp16.onnx"), ec) && std::filesystem::exists(dir / (defaultModel(3) + "_fp16.onnx"), ec);
}

bool RifeDmlBackend::initialize(ID3D12Device* device, ID3D12CommandQueue* queue, uint32_t width, uint32_t height, DXGI_FORMAT format, unsigned multiplier) {
    shutdown();
    auto& p = *p_;
    p.rt = &runtime();
    if (!p.rt->api) { log::error("rife", "runtime unavailable: " + p.rt->error); return false; }
    if (format != DXGI_FORMAT_R8G8B8A8_UNORM) { log::error("rife", std::format("SDR colour only (format {})", int(format))); return false; }
    if (multiplier < 2 || multiplier > 3) { log::error("rife", std::format("multiplier {} not supported (2X or 3X)", multiplier)); return false; }
    p.device = device; p.width = width; p.height = height; p.model = selectedModel(multiplier);
    p.light = lightDivisor();
    p.flowW = p.light ? std::max(32u, (width + p.light / 2) / p.light) : width;
    p.flowH = p.light ? std::max(32u, (height + p.light / 2) / p.light) : height;
    const uint32_t pad = padMultiple(p.model);
    p.pw = padded(p.flowW, pad); p.ph = padded(p.flowH, pad);
    p.multiplier = multiplier; p.format = format;
    const auto modelPath = runtimeDirectory() / (p.model + (p.light ? "_flow_fp16.onnx" : "_fp16.onnx"));
    if (FAILED(p.rt->createDevice(device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&p.dmlDevice)))) { log::error("rife", "DMLCreateDevice failed"); return false; }
    const auto* api = p.rt->api;
    if (!p.check(api->CreateSessionOptions(&p.options), "CreateSessionOptions") ||
        !p.check(api->DisableMemPattern(p.options), "DisableMemPattern") ||
        !p.check(api->SetSessionExecutionMode(p.options, ORT_SEQUENTIAL), "SetSessionExecutionMode") ||
        !p.check(api->SetSessionGraphOptimizationLevel(p.options, ORT_ENABLE_ALL), "SetSessionGraphOptimizationLevel") ||
        !p.check(api->AddFreeDimensionOverrideByName(p.options, "height", int64_t(p.ph)), "height override") ||
        !p.check(api->AddFreeDimensionOverrideByName(p.options, "width", int64_t(p.pw)), "width override") ||
        // The graph's own queue: ONNX Runtime records and submits there, so queue
        // order places each Run between our pack and unpack lists.
        !p.check(p.rt->dml->SessionOptionsAppendExecutionProvider_DML1(p.options, p.dmlDevice.Get(), queue), "AppendExecutionProvider_DML1"))
        return false;
    const auto started = std::chrono::steady_clock::now();
    if (!p.check(api->CreateSession(p.rt->env, modelPath.c_str(), p.options, &p.session), "CreateSession")) return false;
    const double sessionMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    if (!p.check(api->CreateMemoryInfo("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault, &p.memory), "CreateMemoryInfo")) return false;
    for (auto& t : p.planes) if (!p.makeTensor(t, 3)) return false;
    p.times.resize(p.steps()); p.outputs.resize(p.steps());
    for (auto& t : p.times) if (!p.makeTensor(t, 1)) return false;
    for (auto& t : p.outputs) if (!p.makeTensor(t, p.light ? 4 : 3)) return false;
    if (p.light) {
        p.masks.resize(p.steps());
        for (auto& t : p.masks) if (!p.makeTensor(t, 1)) return false;
    }
    for (unsigned cur = 0; cur < 2; ++cur)
        for (unsigned s = 0; s < p.steps(); ++s) {
            OrtIoBinding* b = nullptr;
            if (!p.check(api->CreateIoBinding(p.session, &b), "CreateIoBinding")) return false;
            p.bindings.push_back(b);
            if (!p.check(api->BindInput(b, "img0", p.planes[1 - cur].value), "BindInput img0") ||
                !p.check(api->BindInput(b, "img1", p.planes[cur].value), "BindInput img1") ||
                !p.check(api->BindInput(b, "timestep", p.times[s].value), "BindInput timestep") ||
                !p.check(api->BindOutput(b, p.light ? "flow" : "output", p.outputs[s].value), "BindOutput") ||
                (p.light && !p.check(api->BindOutput(b, "mask", p.masks[s].value), "BindOutput mask")))
                return false;
        }
    if (p.light) {
        for (unsigned i = 0; i < 2; ++i) {
            p.frames[i] = pipeline::makeTexture(device, width, height, format, false);
            if (!p.frames[i]) return false;
            p.frameStates[i] = D3D12_RESOURCE_STATE_COMMON;
        }
    } else {
        p.input = pipeline::makeTexture(device, width, height, format, false);
        if (!p.input) return false;
        p.inputState = D3D12_RESOURCE_STATE_COMMON;
    }
    std::vector<uint8_t> cs;
    if (!p.pack.loadShader("RifePack.dxil", cs) || !p.pack.create(device, cs, 4 + p.steps(), 1, 1) ||
        !p.unpack.loadShader("RifeUnpack.dxil", cs) || !p.unpack.create(device, cs, 1 + 2 * 3 * 8, 1, 2) ||
        !p.shot.loadShader("RifeShotChange.dxil", cs) || !p.shot.create(device, cs, 7, 1, 3, 8))
        return false;
    if (p.light && (!p.lightPack.loadShader("RifeLightPack.dxil", cs) || !p.lightPack.create(device, cs, 4, 1, 1) ||
                    !p.warp.loadShader("RifeLightWarp.dxil", cs) || !p.warp.create(device, cs, 4 + 8 * 3 * 3, 2, 3)))
        return false;
    {
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rd{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 256, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
        if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&p.cut)))) return false;
        p.cutState = D3D12_RESOURCE_STATE_COMMON;
        constexpr UINT workWords = 2 * 128 * 72 + 32;
        rd.Width = workWords * 4;
        if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&p.work)))) return false;
        D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
        u.Format = DXGI_FORMAT_R32_TYPELESS; u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; u.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        // Shot heap: [0] null SRV, then per planes index: that frame's planes, work, flags.
        for (unsigned cur = 0; cur < 2; ++cur) {
            p.rawUav(p.planes[cur].buffer.Get(), 3, p.cpu(p.shot, 1 + cur * 3));
            u.Buffer.NumElements = workWords;
            device->CreateUnorderedAccessView(p.work.Get(), nullptr, &u, p.cpu(p.shot, 2 + cur * 3));
            u.Buffer.NumElements = 64;
            device->CreateUnorderedAccessView(p.cut.Get(), nullptr, &u, p.cpu(p.shot, 3 + cur * 3));
        }
    }
    pipeline::makeSrv(device, p.input.Get(), format, p.cpu(p.pack, 0));
    p.rawUav(p.planes[0].buffer.Get(), 3, p.cpu(p.pack, 1));
    p.rawUav(p.planes[1].buffer.Get(), 3, p.cpu(p.pack, 2));
    D3D12_SHADER_RESOURCE_VIEW_DESC nullSrv{};
    nullSrv.Format = format; nullSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    nullSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; nullSrv.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(nullptr, &nullSrv, p.cpu(p.pack, 3));
    device->CreateShaderResourceView(nullptr, &nullSrv, p.cpu(p.unpack, 0));
    device->CreateShaderResourceView(nullptr, &nullSrv, p.cpu(p.shot, 0));
    for (unsigned i = 0; i < p.times.size(); ++i) p.rawUav(p.times[i].buffer.Get(), 1, p.cpu(p.pack, 4 + i));
    if (p.light) {
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 2; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&p.staging)))) return false;
        const UINT increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        for (unsigned i = 0; i < 2; ++i)
            pipeline::makeSrv(device, p.frames[i].Get(), format, {p.staging->GetCPUDescriptorHandleForHeapStart().ptr + uint64_t(i) * increment});
        for (unsigned i = 0; i < 2; ++i) {
            p.copyStagedSrv(i, p.lightPack, i);
            p.rawUav(p.planes[i].buffer.Get(), 3, p.cpu(p.lightPack, 2 + i));
            p.copyStagedSrv(1 - i, p.warp, 2 * i);   // previous
            p.copyStagedSrv(i, p.warp, 2 * i + 1);   // current
        }
    }
    p.ok = true;
    if (p.light)
        log::info("rife", std::format("ready model={} light 1/{}: flow {}x{} tensor {}x{}, warp and blend at {}x{} {}X session={:.0f} ms (VEYRA_TEST_RIFE_LIGHT)",
                                      p.model, p.light, p.flowW, p.flowH, p.pw, p.ph, width, height, multiplier, sessionMs));
    else
        log::info("rife", std::format("ready model={} {}x{} tensor {}x{} {}X session={:.0f} ms", p.model, width, height, p.pw, p.ph, multiplier, sessionMs));
    return true;
}

bool RifeDmlBackend::capture(ID3D12GraphicsCommandList* list, ID3D12Resource* color) {
    auto& p = *p_;
    if (!p.ok) return false;
    auto barrier = [&](D3D12_RESOURCE_STATES to) {
        if (p.inputState == to) return;
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = p.input.Get(); b.Transition.StateBefore = p.inputState; b.Transition.StateAfter = to;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        list->ResourceBarrier(1, &b);
        p.inputState = to;
    };
    const unsigned next = 1 - p.current;
    if (p.light) {
        // The full-size frame is kept for the warp; the model gets its area average.
        Impl::transition(list, p.frames[next].Get(), p.frameStates[next], D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyResource(p.frames[next].Get(), color);
        Impl::transition(list, p.frames[next].Get(), p.frameStates[next], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        const uint32_t c[8] = {p.width, p.height, p.flowW, p.flowH, p.pw, p.ph, 0, 0};
        p.dispatch(list, p.lightPack, c, p.gpu(p.lightPack, next), p.gpu(p.lightPack, 2 + next), p.pw, p.ph);
    } else {
        barrier(D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyResource(p.input.Get(), color);
        barrier(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        const uint32_t c[8] = {p.width, p.height, p.pw, p.ph, 0, 0, 0, 0};
        p.dispatch(list, p.pack, c, p.gpu(p.pack, 0), p.gpu(p.pack, 1 + next), p.pw, p.ph);
    }
    if (!p.timesFilled) {
        for (unsigned s = 0; s < p.steps(); ++s) {
            const float t = p.time(s);
            uint32_t f[8] = {p.width, p.height, p.pw, p.ph, 1, 0, 1, 0};
            std::memcpy(&f[5], &t, 4);
            p.dispatch(list, p.pack, f, p.gpu(p.pack, 3), p.gpu(p.pack, 4 + s), p.pw, p.ph);
        }
        p.timesFilled = true;
    }
    // Shot change between the previous and this frame, from the planes just written.
    D3D12_RESOURCE_BARRIER planesWritten{};
    planesWritten.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; planesWritten.UAV.pResource = p.planes[next].buffer.Get();
    list->ResourceBarrier(1, &planesWritten);
    Impl::transition(list, p.cut.Get(), p.cutState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // Median best-match error above 0.07: continuous 30 fps GTA pairs stay below
    // 0.043, cuts reach 0.12-0.13 (RifeShotChange.hlsl).
    const float threshold = 0.07f;
    D3D12_RESOURCE_BARRIER workWritten{};
    workWritten.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; workWritten.UAV.pResource = p.work.Get();
    for (uint32_t mode = 0; mode < 3; ++mode) {
        uint32_t sc[8] = {p.flowW, p.flowH, p.pw, p.ph, mode, next, p.haveFrame ? 1u : 0u, 0};
        std::memcpy(&sc[7], &threshold, 4);
        p.shot.bind(list, reinterpret_cast<const float*>(sc), p.gpu(p.shot, 0).ptr, p.gpu(p.shot, 1 + next * 3).ptr);
        // mode 1 needs both luma pictures; on a first frame it still runs, compare=0 discards it.
        list->Dispatch(mode == 0 ? (128 * 72 + 255) / 256 : mode == 1 ? 32 : 1, 1, 1);
        list->ResourceBarrier(1, &workWritten);
    }
    Impl::transition(list, p.cut.Get(), p.cutState, D3D12_RESOURCE_STATE_PREDICATION);
    p.current = next;
    p.havePrevious = p.haveFrame;
    p.haveFrame = true;
    return true;
}

bool RifeDmlBackend::generate(unsigned sub, unsigned of) {
    auto& p = *p_;
    const int s = p.step(sub, of);
    if (!p.ok || !p.havePrevious || s < 0) return false;
    return p.check(p.rt->api->RunWithBinding(p.session, nullptr, p.bindings[p.current * p.steps() + unsigned(s)]), "RunWithBinding");
}

bool RifeDmlBackend::copyOutput(ID3D12GraphicsCommandList* list, ID3D12Resource* output, unsigned sub, unsigned of) {
    auto& p = *p_;
    const int s = p.step(sub, of);
    if (!p.ok || s < 0) return false;
    const auto key = output;
    auto it = p.outputSlots.find(key);
    UINT slot = 0;
    if (it == p.outputSlots.end()) {
        // Owner textures are fixed for the backend's life; each gets one pair of
        // descriptors per subframe, written once (never while the GPU reads them).
        const UINT index = UINT(p.outputSlots.size());
        if (index >= 8) { log::error("rife", "too many output textures"); return false; }
        p.outputSlots.emplace(key, index);
        slot = index;
        for (unsigned k = 0; k < p.steps(); ++k) {
            if (p.light) {
                const UINT base = 4 + (slot * 3 + k) * 3;
                p.rawUav(p.outputs[k].buffer.Get(), 4, p.cpu(p.warp, base));
                p.rawUav(p.masks[k].buffer.Get(), 1, p.cpu(p.warp, base + 1));
                pipeline::makeUav(p.device, output, p.format, p.cpu(p.warp, base + 2));
                continue;
            }
            const UINT base = 1 + (slot * 3 + k) * 2;
            p.rawUav(p.outputs[k].buffer.Get(), 3, p.cpu(p.unpack, base));
            pipeline::makeUav(p.device, output, p.format, p.cpu(p.unpack, base + 1));
        }
    } else slot = it->second;
    // Across a shot change the network would morph two unrelated pictures: the
    // conversion is skipped and the subframe shows the new frame instead.
    list->SetPredication(p.cut.Get(), 0, D3D12_PREDICATION_OP_NOT_EQUAL_ZERO);   // skipped on a cut
    if (p.light) {
        const uint32_t c[8] = {p.width, p.height, p.flowW, p.flowH, p.pw, p.ph, 0, 0};
        p.dispatch(list, p.warp, c, p.gpu(p.warp, 2 * p.current), p.gpu(p.warp, 4 + (slot * 3 + unsigned(s)) * 3), p.width, p.height);
    } else {
        const uint32_t c[8] = {p.width, p.height, p.pw, p.ph, 0, 0, 0, 0};
        p.dispatch(list, p.unpack, c, p.gpu(p.unpack, 0), p.gpu(p.unpack, 1 + (slot * 3 + unsigned(s)) * 2), p.width, p.height);
    }
    ID3D12Resource* newest = p.light ? p.frames[p.current].Get() : p.input.Get();
    D3D12_RESOURCE_STATES& newestState = p.light ? p.frameStates[p.current] : p.inputState;
    D3D12_RESOURCE_STATES outputState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    Impl::transition(list, output, outputState, D3D12_RESOURCE_STATE_COPY_DEST);
    Impl::transition(list, newest, newestState, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->SetPredication(p.cut.Get(), 0, D3D12_PREDICATION_OP_EQUAL_ZERO);       // skipped unless a cut
    list->CopyResource(output, newest);
    list->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
    Impl::transition(list, newest, newestState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Impl::transition(list, output, outputState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    return true;
}

void RifeDmlBackend::reset() { p_->haveFrame = p_->havePrevious = false; }
bool RifeDmlBackend::hasPair() const { return p_->havePrevious; }
bool RifeDmlBackend::ready() const { return p_->ok; }
const std::string& RifeDmlBackend::modelName() const { return p_->model; }

void RifeDmlBackend::shutdown() {
    auto& p = *p_;
    if (!p.rt || !p.rt->api) { p = Impl{}; return; }
    const auto* api = p.rt->api;
    for (auto* b : p.bindings) api->ReleaseIoBinding(b);
    p.bindings.clear();
    for (auto& t : p.planes) p.release(t);
    for (auto& t : p.times) p.release(t);
    for (auto& t : p.outputs) p.release(t);
    for (auto& t : p.masks) p.release(t);
    if (p.session) api->ReleaseSession(p.session);
    if (p.options) api->ReleaseSessionOptions(p.options);
    if (p.memory) api->ReleaseMemoryInfo(p.memory);
    const Runtime* rt = p.rt;
    p = Impl{};
    p.rt = rt;
}
namespace {
// XeSS motion from RIFE's flow: rife_v4.26 unless VEYRA_TEST_FLOW_RIFE_MODEL names
// another flow model built by make_rife_flow_models.py (file <name>_flow_fp16.onnx).
std::string flowModel() {
    wchar_t value[64]{};
    if (GetEnvironmentVariableW(L"VEYRA_TEST_FLOW_RIFE_MODEL", value, 64)) {
        std::string name;
        for (const wchar_t* c = value; *c; ++c) name.push_back(char(*c));
        return name;
    }
    return "rife_v4.26";
}
}

struct RifeFlow::Impl {
    const Runtime* rt = nullptr;
    ID3D12Device* device = nullptr;
    ComPtr<IDMLDevice> dmlDevice;
    OrtSessionOptions* options = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* memory = nullptr;
    OrtIoBinding* binding = nullptr;
    struct Tensor { ComPtr<ID3D12Resource> buffer; void* allocation = nullptr; OrtValue* value = nullptr; };
    Tensor img0, img1, timestep, flow, mask;
    uint32_t inputW = 0, inputH = 0, motionW = 0, motionH = 0, flowW = 0, flowH = 0, pw = 0, ph = 0;
    // Pack heap: [0] previous SRV, [1] current SRV, [2] img0, [3] img1, [4] timestep UAVs.
    // Convert heap: [0] null SRV, [1] flow tensor UAV, [2] motion UAV (raw UAV with edges).
    pipeline::ComputePass pack, convert;
    ComPtr<ID3D12DescriptorHeap> staging;   // SRVs are created here and copied (see DescriptorStager)
    std::string model;
    bool ok = false;
    // Full-size motion edges (enableEdges, VEYRA_TEST_FLOW_RIFE_EDGE): RifeFlowToMotion writes raw,
    // RifeMotionEdge chooses (chosen, flags), RifeMotionEdgeFilter writes the motion texture.
    // Edge heap: [0] raw, [1] current, [2] previous, [3] camera histogram SRVs, [4] chosen, [5] flags UAVs.
    // Filter heap: [0] raw, [1] chosen, [2] flags SRVs, [3] motion UAV.
    bool edges = false;
    float globalScale = 0.f;
    ID3D12Resource* previousColor = nullptr;   // owned by the graph
    ID3D12Resource* currentColor = nullptr;
    ID3D12Resource* motionTexture = nullptr;
    ComPtr<ID3D12Resource> raw, chosen, flags;
    std::array<D3D12_RESOURCE_STATES, 3> edgeStates{};
    pipeline::ComputePass edge, edgeFilter;
    ComPtr<ID3D12DescriptorHeap> edgeStaging;

    bool check(OrtStatus* s, const char* what) {
        if (!s) return true;
        log::error("rife-flow", std::format("{} failed: {}", what, rt->api->GetErrorMessage(s)));
        rt->api->ReleaseStatus(s);
        return false;
    }
    bool makeTensor(Tensor& t, int64_t channels) {
        const uint64_t bytes = uint64_t(channels) * pw * ph * 4;
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC rd{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
        if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&t.buffer)))) {
            log::error("rife-flow", std::format("tensor buffer {} bytes failed", bytes));
            return false;
        }
        if (!check(rt->dml->CreateGPUAllocationFromD3DResource(t.buffer.Get(), &t.allocation), "CreateGPUAllocationFromD3DResource")) return false;
        const int64_t shape[4] = {1, channels, int64_t(ph), int64_t(pw)};
        return check(rt->api->CreateTensorWithDataAsOrtValue(memory, t.allocation, size_t(bytes), shape, 4,
                                                             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t.value), "CreateTensor");
    }
    void release(Tensor& t) {
        if (t.value) rt->api->ReleaseValue(t.value);
        if (t.allocation) rt->dml->FreeGPUAllocation(t.allocation);
        t = {};
    }
    static D3D12_CPU_DESCRIPTOR_HANDLE cpu(ID3D12DescriptorHeap* heap, UINT increment, UINT slot) {
        return {heap->GetCPUDescriptorHandleForHeapStart().ptr + uint64_t(slot) * increment};
    }
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(const pipeline::ComputePass& pass, UINT slot) const {
        return {pass.heap->GetGPUDescriptorHandleForHeapStart().ptr + uint64_t(slot) * pass.increment};
    }
    void rawUav(ID3D12Resource* buffer, uint64_t channels, D3D12_CPU_DESCRIPTOR_HANDLE handle) const {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
        u.Format = DXGI_FORMAT_R32_TYPELESS;
        u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        u.Buffer.NumElements = UINT(channels * pw * ph);
        u.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device->CreateUnorderedAccessView(buffer, nullptr, &u, handle);
    }
    // SRV created in the non-visible staging heap, then copied into the pass heap.
    void stagedSrv(ID3D12Resource* texture, DXGI_FORMAT format, UINT stagingSlot, const pipeline::ComputePass& pass, UINT slot) {
        const UINT increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_SHADER_RESOURCE_VIEW_DESC v{};
        v.Format = format; v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; v.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(texture, &v, cpu(staging.Get(), increment, stagingSlot));
        device->CopyDescriptorsSimple(1, cpu(pass.heap.Get(), pass.increment, slot), cpu(staging.Get(), increment, stagingSlot),
                                      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
};

RifeFlow::RifeFlow() : p_(std::make_unique<Impl>()) {}
RifeFlow::~RifeFlow() { shutdown(); }

bool RifeFlow::initialize(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Resource* previous, ID3D12Resource* current,
                          ID3D12Resource* motion, unsigned divisor) {
    shutdown();
    auto& p = *p_;
    p.rt = &runtime();
    if (!p.rt->api) { log::error("rife-flow", "runtime unavailable: " + p.rt->error); return false; }
    if (!device || !queue || !previous || !current || !motion || divisor < 2 || divisor > 8) { log::error("rife-flow", "invalid arguments"); return false; }
    const auto in = current->GetDesc(), prev = previous->GetDesc(), out = motion->GetDesc();
    if (in.Format != DXGI_FORMAT_B8G8R8A8_UNORM || prev.Format != in.Format || prev.Width != in.Width || prev.Height != in.Height ||
        out.Format != DXGI_FORMAT_R16G16_FLOAT) {
        log::error("rife-flow", "unexpected input or motion texture");
        return false;
    }
    p.device = device;
    p.previousColor = previous; p.currentColor = current; p.motionTexture = motion;
    p.inputW = uint32_t(in.Width); p.inputH = in.Height;
    p.motionW = uint32_t(out.Width); p.motionH = out.Height;
    p.flowW = std::max(32u, (p.motionW + divisor / 2) / divisor); p.flowH = std::max(32u, (p.motionH + divisor / 2) / divisor);
    p.model = flowModel();
    const uint32_t pad = padMultiple(p.model);
    p.pw = padded(p.flowW, pad); p.ph = padded(p.flowH, pad);
    const auto modelPath = runtimeDirectory() / (p.model + "_flow_fp16.onnx");
    std::error_code ec;
    if (!std::filesystem::exists(modelPath, ec)) { log::error("rife-flow", "model missing: " + modelPath.string()); return false; }
    if (FAILED(p.rt->createDevice(device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&p.dmlDevice)))) { log::error("rife-flow", "DMLCreateDevice failed"); return false; }
    const auto* api = p.rt->api;
    if (!p.check(api->CreateSessionOptions(&p.options), "CreateSessionOptions") ||
        !p.check(api->DisableMemPattern(p.options), "DisableMemPattern") ||
        !p.check(api->SetSessionExecutionMode(p.options, ORT_SEQUENTIAL), "SetSessionExecutionMode") ||
        !p.check(api->SetSessionGraphOptimizationLevel(p.options, ORT_ENABLE_ALL), "SetSessionGraphOptimizationLevel") ||
        !p.check(api->AddFreeDimensionOverrideByName(p.options, "height", int64_t(p.ph)), "height override") ||
        !p.check(api->AddFreeDimensionOverrideByName(p.options, "width", int64_t(p.pw)), "width override") ||
        !p.check(p.rt->dml->SessionOptionsAppendExecutionProvider_DML1(p.options, p.dmlDevice.Get(), queue), "AppendExecutionProvider_DML1"))
        return false;
    const auto started = std::chrono::steady_clock::now();
    if (!p.check(api->CreateSession(p.rt->env, modelPath.c_str(), p.options, &p.session), "CreateSession")) return false;
    const double sessionMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    if (!p.check(api->CreateMemoryInfo("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault, &p.memory), "CreateMemoryInfo")) return false;
    if (!p.makeTensor(p.img0, 3) || !p.makeTensor(p.img1, 3) || !p.makeTensor(p.timestep, 1) || !p.makeTensor(p.flow, 4) || !p.makeTensor(p.mask, 1)) return false;
    if (!p.check(api->CreateIoBinding(p.session, &p.binding), "CreateIoBinding") ||
        !p.check(api->BindInput(p.binding, "img0", p.img0.value), "BindInput img0") ||
        !p.check(api->BindInput(p.binding, "img1", p.img1.value), "BindInput img1") ||
        !p.check(api->BindInput(p.binding, "timestep", p.timestep.value), "BindInput timestep") ||
        !p.check(api->BindOutput(p.binding, "flow", p.flow.value), "BindOutput flow") ||
        !p.check(api->BindOutput(p.binding, "mask", p.mask.value), "BindOutput mask"))
        return false;
    std::vector<uint8_t> cs;
    if (!p.pack.loadShader("RifeFlowPack.dxil", cs) || !p.pack.create(device, cs, 5, 2, 3) ||
        !p.convert.loadShader("RifeFlowToMotion.dxil", cs) || !p.convert.create(device, cs, 3, 1, 2))
        return false;
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 3; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&p.staging)))) return false;
    p.stagedSrv(previous, in.Format, 0, p.pack, 0);
    p.stagedSrv(current, in.Format, 1, p.pack, 1);
    p.stagedSrv(nullptr, in.Format, 2, p.convert, 0);
    p.rawUav(p.img0.buffer.Get(), 3, Impl::cpu(p.pack.heap.Get(), p.pack.increment, 2));
    p.rawUav(p.img1.buffer.Get(), 3, Impl::cpu(p.pack.heap.Get(), p.pack.increment, 3));
    p.rawUav(p.timestep.buffer.Get(), 1, Impl::cpu(p.pack.heap.Get(), p.pack.increment, 4));
    p.rawUav(p.flow.buffer.Get(), 4, Impl::cpu(p.convert.heap.Get(), p.convert.increment, 1));
    pipeline::makeUav(device, motion, out.Format, Impl::cpu(p.convert.heap.Get(), p.convert.increment, 2));
    p.ok = true;
    log::info("rife-flow", std::format("ready model={} input={}x{} flow={}x{} tensor {}x{} motion={}x{} session={:.0f} ms",
                                       p.model, p.inputW, p.inputH, p.flowW, p.flowH, p.pw, p.ph, p.motionW, p.motionH, sessionMs));
    return true;
}

bool RifeFlow::pack(ID3D12GraphicsCommandList* list) {
    auto& p = *p_;
    if (!p.ok || !list) return false;
    const float t = 0.5f;
    uint32_t c[8] = {p.inputW, p.inputH, p.flowW, p.flowH, p.pw, p.ph, 0, 0};
    std::memcpy(&c[6], &t, 4);
    p.pack.bind(list, reinterpret_cast<const float*>(c), p.gpu(p.pack, 0).ptr, p.gpu(p.pack, 2).ptr);
    list->Dispatch((p.pw + 15) / 16, (p.ph + 15) / 16, 1);
    D3D12_RESOURCE_BARRIER b[3]{};
    ID3D12Resource* written[3] = {p.img0.buffer.Get(), p.img1.buffer.Get(), p.timestep.buffer.Get()};
    for (int i = 0; i < 3; ++i) { b[i].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; b[i].UAV.pResource = written[i]; }
    list->ResourceBarrier(3, b);
    return true;
}

bool RifeFlow::run() {
    auto& p = *p_;
    return p.ok && p.check(p.rt->api->RunWithBinding(p.session, nullptr, p.binding), "RunWithBinding");
}

bool RifeFlow::enableEdges(ID3D12Resource* globalMotion, float globalScale) {
    auto& p = *p_;
    if (!p.ok) return false;
    p.raw = pipeline::makeTexture(p.device, p.motionW, p.motionH, DXGI_FORMAT_R16G16_FLOAT, true);
    p.chosen = pipeline::makeTexture(p.device, p.motionW, p.motionH, DXGI_FORMAT_R16G16_FLOAT, true);
    p.flags = pipeline::makeTexture(p.device, p.motionW, p.motionH, DXGI_FORMAT_R8_UINT, true);
    if (!p.raw || !p.chosen || !p.flags) return false;
    p.edgeStates.fill(D3D12_RESOURCE_STATE_COMMON);
    std::vector<uint8_t> cs;
    if (!p.edge.loadShader("RifeMotionEdge.dxil", cs) || !p.edge.create(p.device, cs, 6, 4, 2, 8, 0, true) ||
        !p.edgeFilter.loadShader("RifeMotionEdgeFilter.dxil", cs) || !p.edgeFilter.create(p.device, cs, 4, 3, 1))
        return false;
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 7; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(p.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&p.edgeStaging)))) return false;
    const UINT increment = p.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    UINT next = 0;
    auto srv = [&](ID3D12Resource* texture, DXGI_FORMAT format, const pipeline::ComputePass& pass, UINT slot) {
        const auto staged = Impl::cpu(p.edgeStaging.Get(), increment, next++);
        pipeline::makeSrv(p.device, texture, format, staged);
        p.device->CopyDescriptorsSimple(1, Impl::cpu(pass.heap.Get(), pass.increment, slot), staged, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    };
    const auto colour = DXGI_FORMAT_B8G8R8A8_UNORM;
    srv(p.raw.Get(), DXGI_FORMAT_R16G16_FLOAT, p.edge, 0);
    srv(p.currentColor, colour, p.edge, 1);
    srv(p.previousColor, colour, p.edge, 2);
    srv(globalMotion, DXGI_FORMAT_R32_UINT, p.edge, 3);   // null view without a camera estimate
    pipeline::makeUav(p.device, p.chosen.Get(), DXGI_FORMAT_R16G16_FLOAT, Impl::cpu(p.edge.heap.Get(), p.edge.increment, 4));
    pipeline::makeUav(p.device, p.flags.Get(), DXGI_FORMAT_R8_UINT, Impl::cpu(p.edge.heap.Get(), p.edge.increment, 5));
    srv(p.raw.Get(), DXGI_FORMAT_R16G16_FLOAT, p.edgeFilter, 0);
    srv(p.chosen.Get(), DXGI_FORMAT_R16G16_FLOAT, p.edgeFilter, 1);
    srv(p.flags.Get(), DXGI_FORMAT_R8_UINT, p.edgeFilter, 2);
    pipeline::makeUav(p.device, p.motionTexture, DXGI_FORMAT_R16G16_FLOAT, Impl::cpu(p.edgeFilter.heap.Get(), p.edgeFilter.increment, 3));
    // RifeFlowToMotion now writes raw.
    pipeline::makeUav(p.device, p.raw.Get(), DXGI_FORMAT_R16G16_FLOAT, Impl::cpu(p.convert.heap.Get(), p.convert.increment, 2));
    p.globalScale = globalMotion ? globalScale : 0.f;
    p.edges = true;
    return true;
}

bool RifeFlow::edges() const { return p_->edges; }

bool RifeFlow::convert(ID3D12GraphicsCommandList* list) {
    auto& p = *p_;
    if (!p.ok || !list) return false;
    auto to = [&](int i, D3D12_RESOURCE_STATES state) {
        ID3D12Resource* r[3] = {p.raw.Get(), p.chosen.Get(), p.flags.Get()};
        if (p.edgeStates[i] == state) return;
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = r[i]; b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = p.edgeStates[i]; b.Transition.StateAfter = state;
        list->ResourceBarrier(1, &b);
        p.edgeStates[i] = state;
    };
    const uint32_t groupsX = (p.motionW + 15) / 16, groupsY = (p.motionH + 15) / 16;
    if (p.edges) to(0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // Three iterations: enough on 2026-11-27 captures (offline emulation, shader_sim.py).
    const uint32_t c[8] = {p.motionW, p.motionH, p.flowW, p.flowH, p.pw, p.ph, 3, 0};
    p.convert.bind(list, reinterpret_cast<const float*>(c), p.gpu(p.convert, 0).ptr, p.gpu(p.convert, 1).ptr);
    list->Dispatch(groupsX, groupsY, 1);
    if (!p.edges) return true;
    // Tuned at 2560x1440 (2026-11-27): 40 px apart, edge 6 px + 8 % of the vector, win at 60 % of
    // the own cost minus .005, match limit .05; 15 of 25 neighbours switched to keep a switch, 10 good
    // ones to fill a hole. Lengths scale with the width.
    const float s = float(p.motionW) / 2560.f;
    const float radius = std::max(8.f, std::round(40.f * s)), edgeBase = 6.f * s;
    to(0, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    to(1, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    to(2, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    uint32_t e[8] = {p.motionW, p.motionH};
    const float ef[6] = {radius, edgeBase, .08f, .6f, .05f, p.globalScale};
    std::memcpy(&e[2], ef, sizeof(ef));
    p.edge.bind(list, reinterpret_cast<const float*>(e), p.gpu(p.edge, 0).ptr, p.gpu(p.edge, 4).ptr);
    list->Dispatch(groupsX, groupsY, 1);
    to(1, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    to(2, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    uint32_t f[8] = {p.motionW, p.motionH, 0, 0, 15, 10, 0, 0};
    const float ff[2] = {edgeBase, .08f};
    std::memcpy(&f[2], ff, sizeof(ff));
    p.edgeFilter.bind(list, reinterpret_cast<const float*>(f), p.gpu(p.edgeFilter, 0).ptr, p.gpu(p.edgeFilter, 3).ptr);
    list->Dispatch(groupsX, groupsY, 1);
    return true;
}

bool RifeFlow::ready() const { return p_->ok; }
const std::string& RifeFlow::modelName() const { return p_->model; }

void RifeFlow::shutdown() {
    auto& p = *p_;
    if (!p.rt || !p.rt->api) { p = Impl{}; return; }
    const auto* api = p.rt->api;
    if (p.binding) api->ReleaseIoBinding(p.binding);
    for (auto* t : {&p.img0, &p.img1, &p.timestep, &p.flow, &p.mask}) p.release(*t);
    if (p.session) api->ReleaseSession(p.session);
    if (p.options) api->ReleaseSessionOptions(p.options);
    if (p.memory) api->ReleaseMemoryInfo(p.memory);
    const Runtime* rt = p.rt;
    p = Impl{};
    p.rt = rt;
}
#else
struct RifeDmlBackend::Impl { std::string model; };
RifeDmlBackend::RifeDmlBackend() : p_(std::make_unique<Impl>()) {}
RifeDmlBackend::~RifeDmlBackend() = default;
bool RifeDmlBackend::runtimeInstalled() { return false; }
bool RifeDmlBackend::initialize(ID3D12Device*, ID3D12CommandQueue*, uint32_t, uint32_t, DXGI_FORMAT, unsigned) {
    log::error("rife", "this build has no ONNX Runtime DirectML headers");
    return false;
}
bool RifeDmlBackend::capture(ID3D12GraphicsCommandList*, ID3D12Resource*) { return false; }
bool RifeDmlBackend::generate(unsigned, unsigned) { return false; }
bool RifeDmlBackend::copyOutput(ID3D12GraphicsCommandList*, ID3D12Resource*, unsigned, unsigned) { return false; }
void RifeDmlBackend::reset() {}
bool RifeDmlBackend::hasPair() const { return false; }
bool RifeDmlBackend::ready() const { return false; }
const std::string& RifeDmlBackend::modelName() const { return p_->model; }
void RifeDmlBackend::shutdown() {}

struct RifeFlow::Impl { std::string model; };
RifeFlow::RifeFlow() : p_(std::make_unique<Impl>()) {}
RifeFlow::~RifeFlow() = default;
bool RifeFlow::initialize(ID3D12Device*, ID3D12CommandQueue*, ID3D12Resource*, ID3D12Resource*, ID3D12Resource*, unsigned) {
    log::error("rife-flow", "this build has no ONNX Runtime DirectML headers");
    return false;
}
bool RifeFlow::pack(ID3D12GraphicsCommandList*) { return false; }
bool RifeFlow::run() { return false; }
bool RifeFlow::convert(ID3D12GraphicsCommandList*) { return false; }
bool RifeFlow::enableEdges(ID3D12Resource*, float) { return false; }
bool RifeFlow::edges() const { return false; }
bool RifeFlow::ready() const { return false; }
const std::string& RifeFlow::modelName() const { return p_->model; }
void RifeFlow::shutdown() {}
#endif
}
