// framegen.cpp
// GPU-only 2x frame generation for 64-bit Direct3D 11 games (vsync-paced version).
//
// Works two ways from the same source:
//   * as a d3d11.dll proxy dropped next to the game's exe, or
//   * as framegen.dll loaded by the bundled inject.exe (for games that ignore proxies).
// Either way it patches the shared IDXGISwapChain::Present / Present1 vtable slots,
// so it sees every swapchain in the process.
//
// Pipeline per real frame (all on the GPU):
//   1. copy backbuffer -> cur, build a (possibly downscaled) luma image of it
//   2. compute: 8x8 block-matching motion estimation on luma (cur vs prev)
//   3. pixel shader: motion-compensated midpoint frame -> mid
//   4. copy mid -> backbuffer, Present   (interpolated frame)
//   5. copy cur -> backbuffer, Present   (real frame)
//
// Compatibility features: typeless/sRGB/BGRA/10-bit/FP16 backbuffers, automatic motion
// resolution scaling, swapchain selection heuristics (ignores launcher/UI swapchains),
// deferred-context state isolation with a full save/restore fallback for
// single-threaded devices, tearing-flag handling, try-lock for multi-threaded presenters.
//
// Not tested on hardware - expect to debug.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <mutex>
#include <string>

#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "user32.lib")

using Microsoft::WRL::ComPtr;

#ifndef DXGI_PRESENT_ALLOW_TEARING
#define DXGI_PRESENT_ALLOW_TEARING 0x00000200UL
#endif

// ---------------------------------------------------------------- globals ---
static HMODULE      g_self = nullptr;
static std::wstring g_dir;
static FILE*        g_log = nullptr;

static bool g_enabled     = true;
static int  g_toggleVK    = VK_F10;
static int  g_forceSync   = 1;     // sync interval for BOTH presents; -1 = use the game's value
static int  g_motionScale = 0;     // 0 = auto, else 1 / 2 / 4

static void Log(const char* fmt, ...)
{
    if (!g_log) return;
    va_list a; va_start(a, fmt);
    vfprintf(g_log, fmt, a);
    va_end(a);
    fputc('\n', g_log);
    fflush(g_log);
}

static void LoadConfig()
{
    std::wstring ini = g_dir + L"framegen.ini";
    g_enabled     = GetPrivateProfileIntW(L"framegen", L"Enabled", 1, ini.c_str()) != 0;
    g_toggleVK    = (int)GetPrivateProfileIntW(L"framegen", L"ToggleKey", VK_F10, ini.c_str());
    g_forceSync   = (int)GetPrivateProfileIntW(L"framegen", L"ForceSyncInterval", 1, ini.c_str());
    g_motionScale = (int)GetPrivateProfileIntW(L"framegen", L"MotionScale", 0, ini.c_str());
    if (g_forceSync > 4) g_forceSync = 4;
    if (g_forceSync < -1) g_forceSync = -1;
    if (g_motionScale != 1 && g_motionScale != 2 && g_motionScale != 4) g_motionScale = 0;
}

// ------------------------------------------------------------------ HLSL ----
static const char* kHLSL = R"HLSL(
cbuffer CB : register(b0)
{
    uint2 gSize;     // full-resolution size
    uint2 gBlocks;   // motion block grid
    uint2 gLSize;    // luma (motion-estimation) image size
    uint  gScale;    // full-res pixels per luma pixel
    uint  gPad;
};

Texture2D<float4>          gPrev  : register(t0);
Texture2D<float4>          gCur   : register(t1);
StructuredBuffer<float4>   gMVIn  : register(t2);
Texture2D<float>           gPrevL : register(t3);
Texture2D<float>           gCurL  : register(t4);
RWStructuredBuffer<float4> gMVOut : register(u0);
SamplerState               gLin   : register(s0);

#define BS   8
#define R    8
#define TILE 24

groupshared float sCur[BS * BS];
groupshared float sPrev[TILE * TILE];
groupshared uint  sKey[128];
groupshared int2  sBest;

float Luma(float3 c) { return min(dot(c, float3(0.299, 0.587, 0.114)), 4.0); }

float Sad(int2 d)
{
    float s = 0;
    [unroll] for (int y = 0; y < BS; ++y)
    [unroll] for (int x = 0; x < BS; ++x)
        s += abs(sCur[y * BS + x] - sPrev[(y + R + d.y) * TILE + (x + R + d.x)]);
    // small bias toward zero motion to suppress noise vectors
    return s + 0.15 * (abs(d.x) + abs(d.y));
}

[numthreads(128, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    int2 sz  = int2(gLSize);
    int2 org = int2(gid.xy) * BS;
    int2 po  = org - R;

    if (gi < 64)
    {
        int2 p = org + int2(gi % 8, gi / 8);
        sCur[gi] = gCurL.Load(int3(clamp(p, int2(0, 0), sz - 1), 0));
    }
    for (uint i = gi; i < TILE * TILE; i += 128)
    {
        int2 p = po + int2(i % TILE, i / TILE);
        sPrev[i] = gPrevL.Load(int3(clamp(p, int2(0, 0), sz - 1), 0));
    }
    GroupMemoryBarrierWithGroupSync();

    // coarse pass: 9x9 candidates, step 2
    uint key = 0xFFFFFFFF;
    if (gi < 81)
    {
        int2 d = int2(gi % 9, gi / 9) * 2 - R;
        key = ((uint)(Sad(d) * 256.0) << 8) | gi;
    }
    sKey[gi] = key;
    GroupMemoryBarrierWithGroupSync();

    if (gi == 0)
    {
        uint best = 0xFFFFFFFF;
        for (uint k = 0; k < 81; ++k) best = min(best, sKey[k]);
        uint bi = best & 255;
        sBest = int2(bi % 9, bi / 9) * 2 - R;
    }
    GroupMemoryBarrierWithGroupSync();

    // refine pass: 3x3 around the coarse winner
    key = 0xFFFFFFFF;
    if (gi < 9)
    {
        int2 d = clamp(sBest + int2(gi % 3, gi / 3) - 1, int2(-R, -R), int2(R, R));
        key = ((uint)(Sad(d) * 256.0) << 8) | gi;
    }
    sKey[gi] = key;
    GroupMemoryBarrierWithGroupSync();

    if (gi == 0)
    {
        uint best = 0xFFFFFFFF;
        for (uint k = 0; k < 9; ++k) best = min(best, sKey[k]);
        uint bi = best & 255;
        int2 d = clamp(sBest + int2(bi % 3, bi / 3) - 1, int2(-R, -R), int2(R, R));
        float sadPx = (float)(best >> 8) / 256.0 / 64.0;
        if (gid.x < gBlocks.x && gid.y < gBlocks.y)
            gMVOut[gid.y * gBlocks.x + gid.x] = float4((float2)d, sadPx, 0);
    }
}

float4 VSMain(uint id : SV_VertexID) : SV_Position
{
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}

// Luma image for motion estimation (box-filtered when downscaling).
float PSLuma(float4 pos : SV_Position) : SV_Target
{
    float2 invSz = 1.0 / float2(gSize);
    float2 c = pos.xy * (float)gScale;
    if (gScale == 1)
        return Luma(gCur.SampleLevel(gLin, c * invSz, 0).rgb);
    float o = (float)gScale * 0.25;
    float s = Luma(gCur.SampleLevel(gLin, (c + float2(-o, -o)) * invSz, 0).rgb)
            + Luma(gCur.SampleLevel(gLin, (c + float2( o, -o)) * invSz, 0).rgb)
            + Luma(gCur.SampleLevel(gLin, (c + float2(-o,  o)) * invSz, 0).rgb)
            + Luma(gCur.SampleLevel(gLin, (c + float2( o,  o)) * invSz, 0).rgb);
    return s * 0.25;
}

// Motion convention: prev(p + d) ~= cur(p), so the midpoint frame is
// mid(x) = 0.5 * (cur(x - d/2) + prev(x + d/2)).  d is stored in luma pixels.
float4 PSMain(float4 pos : SV_Position) : SV_Target
{
    float  sc    = (float)gScale;
    float2 invSz = 1.0 / float2(gSize);
    float2 bp = pos.xy / (BS * sc) - 0.5;
    int2 b0 = (int2)floor(bp);

    float  bestErr = 1e9;
    float3 bestCol = 0;

    [unroll] for (int i = 0; i < 5; ++i)
    {
        float2 d = 0;
        if (i < 4)
        {
            int2 b = clamp(b0 + int2(i & 1, i >> 1), int2(0, 0), int2(gBlocks) - 1);
            d = gMVIn[b.y * gBlocks.x + b.x].xy * sc;
        }
        float3 a = gCur.SampleLevel(gLin, (pos.xy - 0.5 * d) * invSz, 0).rgb;
        float3 c = gPrev.SampleLevel(gLin, (pos.xy + 0.5 * d) * invSz, 0).rgb;
        float err = dot(abs(a - c), float3(0.299, 0.587, 0.114));
        if (i < 4) err += 0.012;               // zero motion wins ties
        if (err < bestErr) { bestErr = err; bestCol = 0.5 * (a + c); }
    }
    return float4(bestCol, 1);
}
)HLSL";

static ComPtr<ID3DBlob> Compile(const char* entry, const char* target)
{
    ComPtr<ID3DBlob> code, err;
    HRESULT hr = D3DCompile(kHLSL, strlen(kHLSL), "framegen.hlsl", nullptr, nullptr,
                            entry, target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &err);
    if (FAILED(hr))
    {
        Log("shader compile failed (%s): %s", entry,
            err ? (const char*)err->GetBufferPointer() : "unknown");
        return nullptr;
    }
    return code;
}

// Maps a backbuffer format to the typed format used for our SRV/RTV views.
static bool ViewFormat(DXGI_FORMAT f, DXGI_FORMAT* out)
{
    switch (f)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
        *out = f; return true;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:    *out = DXGI_FORMAT_R8G8B8A8_UNORM;    return true;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:    *out = DXGI_FORMAT_B8G8R8A8_UNORM;    return true;
    case DXGI_FORMAT_B8G8R8X8_TYPELESS:    *out = DXGI_FORMAT_B8G8R8X8_UNORM;    return true;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: *out = DXGI_FORMAT_R10G10B10A2_UNORM; return true;
    default: return false;
    }
}

template <class T> static void Rel(T*& p) { if (p) { p->Release(); p = nullptr; } }

// Full pipeline-state snapshot, used only when deferred contexts are unavailable.
struct StateBackup
{
    ID3D11InputLayout* il = nullptr; D3D11_PRIMITIVE_TOPOLOGY topo = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
    ID3D11VertexShader* vs = nullptr; ID3D11HullShader* hs = nullptr;
    ID3D11DomainShader* ds = nullptr; ID3D11GeometryShader* gs = nullptr;
    ID3D11PixelShader* ps = nullptr; ID3D11ShaderResourceView* psSrv[3] = {};
    ID3D11SamplerState* psSamp = nullptr; ID3D11Buffer* psCb = nullptr;
    ID3D11ComputeShader* cs = nullptr; ID3D11ShaderResourceView* csSrv[2] = {};
    ID3D11UnorderedAccessView* csUav = nullptr; ID3D11Buffer* csCb = nullptr;
    ID3D11RasterizerState* rs = nullptr;
    UINT nvp = 0; D3D11_VIEWPORT vp[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] = {};
    ID3D11RenderTargetView* rtv[8] = {}; ID3D11DepthStencilView* dsv = nullptr;
    ID3D11BlendState* bs = nullptr; FLOAT bf[4] = {}; UINT sm = 0;
    ID3D11DepthStencilState* dss = nullptr; UINT sref = 0;

    void Save(ID3D11DeviceContext* c)
    {
        c->IAGetInputLayout(&il); c->IAGetPrimitiveTopology(&topo);
        c->VSGetShader(&vs, nullptr, nullptr); c->HSGetShader(&hs, nullptr, nullptr);
        c->DSGetShader(&ds, nullptr, nullptr); c->GSGetShader(&gs, nullptr, nullptr);
        c->PSGetShader(&ps, nullptr, nullptr);
        c->PSGetShaderResources(0, 3, psSrv); c->PSGetSamplers(0, 1, &psSamp);
        c->PSGetConstantBuffers(0, 1, &psCb);
        c->CSGetShader(&cs, nullptr, nullptr);
        c->CSGetShaderResources(3, 2, csSrv); c->CSGetUnorderedAccessViews(0, 1, &csUav);
        c->CSGetConstantBuffers(0, 1, &csCb);
        c->RSGetState(&rs);
        nvp = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE; c->RSGetViewports(&nvp, vp);
        c->OMGetRenderTargets(8, rtv, &dsv);
        c->OMGetBlendState(&bs, bf, &sm);
        c->OMGetDepthStencilState(&dss, &sref);
    }

    void Restore(ID3D11DeviceContext* c)
    {
        c->IASetInputLayout(il); c->IASetPrimitiveTopology(topo);
        c->VSSetShader(vs, nullptr, 0); c->HSSetShader(hs, nullptr, 0);
        c->DSSetShader(ds, nullptr, 0); c->GSSetShader(gs, nullptr, 0);
        c->PSSetShader(ps, nullptr, 0);
        c->PSSetShaderResources(0, 3, psSrv); c->PSSetSamplers(0, 1, &psSamp);
        c->PSSetConstantBuffers(0, 1, &psCb);
        c->CSSetShader(cs, nullptr, 0);
        c->CSSetShaderResources(3, 2, csSrv); c->CSSetUnorderedAccessViews(0, 1, &csUav, nullptr);
        c->CSSetConstantBuffers(0, 1, &csCb);
        c->RSSetState(rs); c->RSSetViewports(nvp, vp);
        c->OMSetRenderTargets(8, rtv, dsv);
        c->OMSetBlendState(bs, bf, sm);
        c->OMSetDepthStencilState(dss, sref);

        Rel(il); Rel(vs); Rel(hs); Rel(ds); Rel(gs); Rel(ps); Rel(psSamp); Rel(psCb);
        Rel(cs); Rel(csUav); Rel(csCb); Rel(rs); Rel(dsv); Rel(bs); Rel(dss);
        for (auto& s : psSrv) Rel(s);
        for (auto& s : csSrv) Rel(s);
        for (auto& r : rtv) Rel(r);
    }
};

// -------------------------------------------------------------- FrameGen ----
struct CBData { UINT w, h, bx, by, lw, lh, scale, pad; };

struct Cand { IDXGISwapChain* sc; UINT area; ULONGLONG last; UINT count; };

struct FrameGen
{
    ComPtr<ID3D11Device>        dev;
    ComPtr<ID3D11DeviceContext> ctx, dctx;
    ComPtr<ID3D11ComputeShader> cs;
    ComPtr<ID3D11VertexShader>  vs;
    ComPtr<ID3D11PixelShader>   ps, psLuma;
    ComPtr<ID3D11SamplerState>  samp;
    ComPtr<ID3D11RasterizerState> rs;
    ComPtr<ID3D11BlendState>    bs;
    ComPtr<ID3D11DepthStencilState> dss;
    ComPtr<ID3D11Buffer>        cb, mv;
    ComPtr<ID3D11Texture2D>     tex[2], mid, luma[2];
    ComPtr<ID3D11ShaderResourceView> srv[2], lumaSRV[2], mvSRV;
    ComPtr<ID3D11RenderTargetView>   midRTV, lumaRTV[2];
    ComPtr<ID3D11UnorderedAccessView> mvUAV;

    bool deferred = true;
    UINT w = 0, h = 0, lw = 0, lh = 0, bx = 0, by = 0, scale = 1;
    DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN, vfmt = DXGI_FORMAT_UNKNOWN;
    int  cur = 0;
    bool havePrev = false;
    bool broken = false;
    IDXGISwapChain* owner = nullptr;
    UINT ownerArea = 0;
    ULONGLONG lastSeen = 0;
    Cand cands[8] = {};

    void ResetSize()
    {
        for (int i = 0; i < 2; ++i)
        { tex[i].Reset(); srv[i].Reset(); luma[i].Reset(); lumaSRV[i].Reset(); lumaRTV[i].Reset(); }
        mid.Reset(); midRTV.Reset(); mv.Reset(); mvSRV.Reset(); mvUAV.Reset(); cb.Reset();
        w = h = lw = lh = bx = by = 0; scale = 1;
        fmt = vfmt = DXGI_FORMAT_UNKNOWN; havePrev = false;
    }

    void ResetAll()
    {
        ResetSize();
        cs.Reset(); vs.Reset(); ps.Reset(); psLuma.Reset(); samp.Reset(); rs.Reset();
        bs.Reset(); dss.Reset(); dctx.Reset(); ctx.Reset(); dev.Reset();
        owner = nullptr; ownerArea = 0; broken = false; cur = 0; deferred = true;
    }

    bool InitDevice(ID3D11Device* d)
    {
        dev = d;
        dev->GetImmediateContext(&ctx);

        // Preferred: record on a deferred context and run it with state restore.
        // Devices created with SINGLETHREADED may refuse; then we save/restore manually.
        deferred = SUCCEEDED(dev->CreateDeferredContext(0, &dctx));
        Log(deferred ? "using deferred context" : "deferred context unavailable, using manual state save/restore");

        auto csb = Compile("CSMain", "cs_5_0");
        auto vsb = Compile("VSMain", "vs_5_0");
        auto psb = Compile("PSMain", "ps_5_0");
        auto plb = Compile("PSLuma", "ps_5_0");
        if (!csb || !vsb || !psb || !plb) return false;

        if (FAILED(dev->CreateComputeShader(csb->GetBufferPointer(), csb->GetBufferSize(), nullptr, &cs))) return false;
        if (FAILED(dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs))) return false;
        if (FAILED(dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &ps))) return false;
        if (FAILED(dev->CreatePixelShader(plb->GetBufferPointer(), plb->GetBufferSize(), nullptr, &psLuma))) return false;

        D3D11_SAMPLER_DESC sd{};
        sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sd.MaxLOD = D3D11_FLOAT32_MAX;
        if (FAILED(dev->CreateSamplerState(&sd, &samp))) return false;

        D3D11_RASTERIZER_DESC rd{};
        rd.FillMode = D3D11_FILL_SOLID;
        rd.CullMode = D3D11_CULL_NONE;
        rd.DepthClipEnable = TRUE;
        if (FAILED(dev->CreateRasterizerState(&rd, &rs))) return false;

        D3D11_BLEND_DESC bd{};
        bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        if (FAILED(dev->CreateBlendState(&bd, &bs))) return false;

        D3D11_DEPTH_STENCIL_DESC dd{};
        dd.DepthEnable = FALSE;
        dd.StencilEnable = FALSE;
        dd.DepthFunc = D3D11_COMPARISON_ALWAYS;
        if (FAILED(dev->CreateDepthStencilState(&dd, &dss))) return false;

        Log("device init ok");
        return true;
    }

    bool InitSize(const D3D11_TEXTURE2D_DESC& bd)
    {
        ResetSize();
        if (!ViewFormat(bd.Format, &vfmt))
        {
            Log("unsupported backbuffer format %d", (int)bd.Format);
            return false;
        }
        w = bd.Width; h = bd.Height; fmt = bd.Format;

        scale = (UINT)g_motionScale;
        if (!scale) scale = (h <= 720) ? 1 : (h <= 1440 ? 2 : 4);
        while (scale > 1 && (w / scale < 64 || h / scale < 64)) scale /= 2;
        lw = (w + scale - 1) / scale; lh = (h + scale - 1) / scale;
        bx = (lw + 7) / 8; by = (lh + 7) / 8;

        // full-res frame copies (same format as the backbuffer so CopyResource is legal)
        D3D11_TEXTURE2D_DESC td{};
        td.Width = w; td.Height = h; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = fmt; td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;

        D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
        sv.Format = vfmt; sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D; sv.Texture2D.MipLevels = 1;
        for (int i = 0; i < 2; ++i)
        {
            if (FAILED(dev->CreateTexture2D(&td, nullptr, &tex[i]))) return false;
            if (FAILED(dev->CreateShaderResourceView(tex[i].Get(), &sv, &srv[i]))) return false;
        }
        td.BindFlags |= D3D11_BIND_RENDER_TARGET;
        if (FAILED(dev->CreateTexture2D(&td, nullptr, &mid))) return false;
        D3D11_RENDER_TARGET_VIEW_DESC rv{};
        rv.Format = vfmt; rv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        if (FAILED(dev->CreateRenderTargetView(mid.Get(), &rv, &midRTV))) return false;

        // luma images for motion estimation
        D3D11_TEXTURE2D_DESC ld = td;
        ld.Width = lw; ld.Height = lh; ld.Format = DXGI_FORMAT_R16_FLOAT;
        ld.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        for (int i = 0; i < 2; ++i)
        {
            if (FAILED(dev->CreateTexture2D(&ld, nullptr, &luma[i]))) return false;
            if (FAILED(dev->CreateShaderResourceView(luma[i].Get(), nullptr, &lumaSRV[i]))) return false;
            if (FAILED(dev->CreateRenderTargetView(luma[i].Get(), nullptr, &lumaRTV[i]))) return false;
        }

        // motion vector buffer
        const UINT n = bx * by;
        D3D11_BUFFER_DESC mb{};
        mb.ByteWidth = n * 16;
        mb.Usage = D3D11_USAGE_DEFAULT;
        mb.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        mb.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        mb.StructureByteStride = 16;
        if (FAILED(dev->CreateBuffer(&mb, nullptr, &mv))) return false;

        D3D11_SHADER_RESOURCE_VIEW_DESC bsv{};
        bsv.Format = DXGI_FORMAT_UNKNOWN;
        bsv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        bsv.Buffer.FirstElement = 0; bsv.Buffer.NumElements = n;
        if (FAILED(dev->CreateShaderResourceView(mv.Get(), &bsv, &mvSRV))) return false;

        D3D11_UNORDERED_ACCESS_VIEW_DESC buv{};
        buv.Format = DXGI_FORMAT_UNKNOWN;
        buv.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        buv.Buffer.FirstElement = 0; buv.Buffer.NumElements = n;
        if (FAILED(dev->CreateUnorderedAccessView(mv.Get(), &buv, &mvUAV))) return false;

        CBData c{ w, h, bx, by, lw, lh, scale, 0 };
        D3D11_BUFFER_DESC cbd{};
        cbd.ByteWidth = sizeof(CBData);
        cbd.Usage = D3D11_USAGE_IMMUTABLE;
        cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        D3D11_SUBRESOURCE_DATA init{ &c, 0, 0 };
        if (FAILED(dev->CreateBuffer(&cbd, &init, &cb))) return false;

        Log("size init %ux%u fmt %d, motion scale %u (%ux%u luma, %u blocks)",
            w, h, (int)fmt, scale, lw, lh, n);
        return true;
    }

    // Decides whether this swapchain is the game's main one, and (re)builds resources.
    bool Accept(IDXGISwapChain* sc, ID3D11Device* d, const D3D11_TEXTURE2D_DESC& bd)
    {
        const ULONGLONG now = GetTickCount64();
        const UINT area = bd.Width * bd.Height;

        // per-swapchain presentation statistics (pointers are only used as identities)
        Cand* c = nullptr; Cand* oldest = &cands[0];
        for (auto& k : cands)
        {
            if (k.sc == sc) { c = &k; break; }
            if (k.last < oldest->last) oldest = &k;
        }
        if (!c) { c = oldest; *c = Cand{ sc, 0, 0, 0 }; }
        if (now - c->last > 1000) c->count = 0;
        c->area = area; c->last = now; c->count++;

        if (owner && owner != sc && now - lastSeen > 2000) ResetAll();   // owner went quiet

        if (owner && owner != sc)
        {
            // switch only to a clearly larger swapchain that has presented steadily
            if (area > ownerArea + ownerArea / 2 && c->count >= 60 && bd.SampleDesc.Count == 1)
                ResetAll();
            else
                return false;
        }
        if (!owner)
        {
            if (area < 320 * 200 || bd.SampleDesc.Count != 1 || c->count < 10) return false;
            owner = sc;
            Log("selected swapchain %p (%ux%u)", (void*)sc, bd.Width, bd.Height);
        }
        ownerArea = area;
        lastSeen = now;
        if (broken) return false;

        if (d != dev.Get())
        {
            ResetSize();
            if (!InitDevice(d)) { broken = true; return false; }
        }
        if (bd.Width != w || bd.Height != h || bd.Format != fmt)
        {
            if (!InitSize(bd)) { broken = true; return false; }
        }
        return true;
    }

    void Exec()
    {
        if (!deferred) return;
        ComPtr<ID3D11CommandList> cl;
        if (SUCCEEDED(dctx->FinishCommandList(FALSE, &cl)))
            ctx->ExecuteCommandList(cl.Get(), TRUE);   // TRUE = restore the game's pipeline state
    }

    // Capture the real frame, update the luma history and (if possible) build the
    // midpoint frame and copy it into the backbuffer.
    void RecordFirst(ID3D11Texture2D* bb, bool interp)
    {
        ID3D11DeviceContext* c = deferred ? dctx.Get() : ctx.Get();
        StateBackup sb;
        if (!deferred) sb.Save(c);

        c->CopyResource(tex[cur].Get(), bb);

        // state shared by both fullscreen passes
        c->IASetInputLayout(nullptr);
        c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        c->VSSetShader(vs.Get(), nullptr, 0);
        c->HSSetShader(nullptr, nullptr, 0);
        c->DSSetShader(nullptr, nullptr, 0);
        c->GSSetShader(nullptr, nullptr, 0);
        c->RSSetState(rs.Get());
        c->OMSetBlendState(bs.Get(), nullptr, 0xFFFFFFFF);
        c->OMSetDepthStencilState(dss.Get(), 0);
        c->PSSetSamplers(0, 1, samp.GetAddressOf());
        c->PSSetConstantBuffers(0, 1, cb.GetAddressOf());

        // --- luma pass (always, so the next frame has a history) ---
        {
            ID3D11RenderTargetView* rtv = lumaRTV[cur].Get();
            D3D11_VIEWPORT vp{ 0.f, 0.f, (float)lw, (float)lh, 0.f, 1.f };
            ID3D11ShaderResourceView* s1 = srv[cur].Get();
            c->OMSetRenderTargets(1, &rtv, nullptr);
            c->RSSetViewports(1, &vp);
            c->PSSetShader(psLuma.Get(), nullptr, 0);
            c->PSSetShaderResources(1, 1, &s1);
            c->Draw(3, 0);
            ID3D11ShaderResourceView* n1 = nullptr;
            c->PSSetShaderResources(1, 1, &n1);
            c->OMSetRenderTargets(0, nullptr, nullptr);
        }

        if (interp)
        {
            const int prev = cur ^ 1;

            // --- motion estimation ---
            ID3D11ShaderResourceView* in[2] = { lumaSRV[prev].Get(), lumaSRV[cur].Get() };
            ID3D11UnorderedAccessView* uav = mvUAV.Get();
            c->CSSetShader(cs.Get(), nullptr, 0);
            c->CSSetShaderResources(3, 2, in);
            c->CSSetConstantBuffers(0, 1, cb.GetAddressOf());
            c->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            c->Dispatch(bx, by, 1);
            ID3D11UnorderedAccessView* nuav = nullptr;
            ID3D11ShaderResourceView* ns2[2] = {};
            c->CSSetUnorderedAccessViews(0, 1, &nuav, nullptr);
            c->CSSetShaderResources(3, 2, ns2);

            // --- motion-compensated midpoint frame ---
            ID3D11RenderTargetView* rtv = midRTV.Get();
            D3D11_VIEWPORT vp{ 0.f, 0.f, (float)w, (float)h, 0.f, 1.f };
            ID3D11ShaderResourceView* pin[3] = { srv[prev].Get(), srv[cur].Get(), mvSRV.Get() };
            c->OMSetRenderTargets(1, &rtv, nullptr);
            c->RSSetViewports(1, &vp);
            c->PSSetShader(ps.Get(), nullptr, 0);
            c->PSSetShaderResources(0, 3, pin);
            c->Draw(3, 0);
            ID3D11ShaderResourceView* ns3[3] = {};
            c->PSSetShaderResources(0, 3, ns3);
            c->OMSetRenderTargets(0, nullptr, nullptr);

            c->CopyResource(bb, mid.Get());
        }

        if (!deferred) sb.Restore(c);
    }

    void CopyRealFrameBack(ID3D11Texture2D* bb)
    {
        ID3D11DeviceContext* c = deferred ? dctx.Get() : ctx.Get();
        c->CopyResource(bb, tex[cur].Get());
        Exec();
    }
};

static FrameGen   g_fg;
static std::mutex g_mtx;

// ----------------------------------------------------------------- hooks ----
typedef HRESULT(STDMETHODCALLTYPE* PresentFn)(IDXGISwapChain*, UINT, UINT);
typedef HRESULT(STDMETHODCALLTYPE* Present1Fn)(IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);
static PresentFn  g_origPresent  = nullptr;
static Present1Fn g_origPresent1 = nullptr;

static thread_local bool t_inHook = false;

static HRESULT DoPresent(IDXGISwapChain* sc, UINT sync, UINT flags,
                         const DXGI_PRESENT_PARAMETERS* pp, bool is1)
{
    auto call = [&](UINT s, UINT f, const DXGI_PRESENT_PARAMETERS* p) -> HRESULT {
        return is1 ? g_origPresent1((IDXGISwapChain1*)sc, s, f, p)
                   : g_origPresent(sc, s, f);
    };

    if (t_inHook || (flags & DXGI_PRESENT_TEST)) return call(sync, flags, pp);

    // toggle hotkey
    static bool keyWasDown = false;
    if (g_toggleVK)
    {
        bool down = (GetAsyncKeyState(g_toggleVK) & 0x8000) != 0;
        if (down && !keyWasDown)
        {
            g_enabled = !g_enabled;
            g_fg.havePrev = false;
            Log("toggled: %s", g_enabled ? "on" : "off");
        }
        keyWasDown = down;
    }
    if (!g_enabled) return call(sync, flags, pp);

    // if another thread is already inside (e.g. a second swapchain), just pass through
    std::unique_lock<std::mutex> lk(g_mtx, std::try_to_lock);
    if (!lk.owns_lock()) return call(sync, flags, pp);

    t_inHook = true;
    struct Guard { ~Guard() { t_inHook = false; } } guard;

    ComPtr<ID3D11Device> d;
    if (FAILED(sc->GetDevice(__uuidof(ID3D11Device), (void**)d.GetAddressOf())))
        return call(sync, flags, pp);                     // not a D3D11 swapchain

    ComPtr<ID3D11Texture2D> bb;
    if (FAILED(sc->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)bb.GetAddressOf())))
        return call(sync, flags, pp);
    D3D11_TEXTURE2D_DESC bd; bb->GetDesc(&bd);

    FrameGen& g = g_fg;
    if (!g.Accept(sc, d.Get(), bd)) return call(sync, flags, pp);

    const bool interp = g.havePrev;
    g.RecordFirst(bb.Get(), interp);
    g.Exec();

    UINT s = (g_forceSync >= 0) ? (UINT)g_forceSync : sync;
    UINT realFlags = flags;
    if (s > 0) realFlags &= ~DXGI_PRESENT_ALLOW_TEARING;

    HRESULT hr = S_OK;
    if (interp)
    {
        hr = call(s, 0, nullptr);                         // interpolated frame, full present
        if (FAILED(hr)) { g.havePrev = false; return hr; }

        bb.Reset();
        if (SUCCEEDED(sc->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)bb.GetAddressOf())))
            g.CopyRealFrameBack(bb.Get());                // put the real frame back
        bb.Reset();
    }
    bb.Reset();                                           // never hold buffer refs past Present
    hr = call(s, realFlags, pp);                          // real frame

    g.cur ^= 1;
    g.havePrev = true;
    return hr;
}

static HRESULT STDMETHODCALLTYPE HookPresent(IDXGISwapChain* sc, UINT sync, UINT flags)
{
    return DoPresent(sc, sync, flags, nullptr, false);
}

static HRESULT STDMETHODCALLTYPE HookPresent1(IDXGISwapChain1* sc, UINT sync, UINT flags,
                                              const DXGI_PRESENT_PARAMETERS* pp)
{
    return DoPresent(sc, sync, flags, pp, true);
}

static bool PatchVtbl(void** vtbl, size_t idx, void* hook, void** orig)
{
    if (vtbl[idx] == hook) return true;                   // already patched (shared vtable)
    DWORD old;
    if (!VirtualProtect(&vtbl[idx], sizeof(void*), PAGE_READWRITE, &old)) return false;
    if (!*orig) *orig = vtbl[idx];
    vtbl[idx] = hook;
    VirtualProtect(&vtbl[idx], sizeof(void*), old, &old);
    return true;
}

// ----------------------------------------------------- real d3d11 loading ---
static HMODULE RealD3D11()
{
    static HMODULE h = [] {
        wchar_t p[MAX_PATH];
        GetSystemDirectoryW(p, MAX_PATH);
        wcscat_s(p, L"\\d3d11.dll");
        return LoadLibraryW(p);
    }();
    return h;
}

// Creates a throwaway device + swapchain from the real d3d11 and patches the shared
// swapchain vtable, so every swapchain in the process goes through our hooks - including
// ones created before this DLL was injected.
static void EnsureHooks()
{
    static std::once_flag once;
    std::call_once(once, [] {
        HMODULE rd = RealD3D11();
        auto create = rd ? (PFN_D3D11_CREATE_DEVICE_AND_SWAP_CHAIN)
                           GetProcAddress(rd, "D3D11CreateDeviceAndSwapChain") : nullptr;
        if (!create) { Log("real d3d11.dll not found"); return; }

        HWND hw = CreateWindowExW(0, L"STATIC", L"fg", WS_OVERLAPPED, 0, 0, 16, 16,
                                  nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
        DXGI_SWAP_CHAIN_DESC sd{};
        sd.BufferDesc.Width = 16; sd.BufferDesc.Height = 16;
        sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = 1;
        sd.OutputWindow = hw;
        sd.Windowed = TRUE;
        sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

        ComPtr<IDXGISwapChain> sc;
        ComPtr<ID3D11Device> dev;
        ComPtr<ID3D11DeviceContext> ctx;
        HRESULT hr = create(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                            D3D11_SDK_VERSION, &sd, &sc, &dev, nullptr, &ctx);
        if (SUCCEEDED(hr))
        {
            void** vt = *(void***)sc.Get();
            PatchVtbl(vt, 8, (void*)&HookPresent, (void**)&g_origPresent);

            ComPtr<IDXGISwapChain1> sc1;
            if (SUCCEEDED(sc.As(&sc1)))
            {
                void** vt1 = *(void***)sc1.Get();
                if (vt1 != vt) PatchVtbl(vt1, 8, (void*)&HookPresent, (void**)&g_origPresent);
                PatchVtbl(vt1, 22, (void*)&HookPresent1, (void**)&g_origPresent1);
            }
            Log("hooks installed (Present=%p Present1=%p)", (void*)g_origPresent, (void*)g_origPresent1);
        }
        else
        {
            Log("dummy device creation failed: 0x%08X", (unsigned)hr);
        }
        sc.Reset(); ctx.Reset(); dev.Reset();
        if (hw) DestroyWindow(hw);
    });
}

static DWORD WINAPI HookThread(LPVOID)
{
    EnsureHooks();      // makes injected (non-proxy) loads work too
    return 0;
}

// -------------------------------------------------------- proxy exports -----
// Only exported when built with framegen.def (the d3d11.dll proxy build).
extern "C" HRESULT WINAPI Proxy_D3D11CreateDevice(
    IDXGIAdapter* adapter, D3D_DRIVER_TYPE type, HMODULE sw, UINT flags,
    const D3D_FEATURE_LEVEL* levels, UINT nLevels, UINT sdk,
    ID3D11Device** dev, D3D_FEATURE_LEVEL* outLevel, ID3D11DeviceContext** ctx)
{
    EnsureHooks();
    auto fn = (PFN_D3D11_CREATE_DEVICE)GetProcAddress(RealD3D11(), "D3D11CreateDevice");
    if (!fn) return E_FAIL;
    return fn(adapter, type, sw, flags, levels, nLevels, sdk, dev, outLevel, ctx);
}

extern "C" HRESULT WINAPI Proxy_D3D11CreateDeviceAndSwapChain(
    IDXGIAdapter* adapter, D3D_DRIVER_TYPE type, HMODULE sw, UINT flags,
    const D3D_FEATURE_LEVEL* levels, UINT nLevels, UINT sdk,
    const DXGI_SWAP_CHAIN_DESC* desc, IDXGISwapChain** sc,
    ID3D11Device** dev, D3D_FEATURE_LEVEL* outLevel, ID3D11DeviceContext** ctx)
{
    EnsureHooks();
    auto fn = (PFN_D3D11_CREATE_DEVICE_AND_SWAP_CHAIN)
              GetProcAddress(RealD3D11(), "D3D11CreateDeviceAndSwapChain");
    if (!fn) return E_FAIL;
    return fn(adapter, type, sw, flags, levels, nLevels, sdk, desc, sc, dev, outLevel, ctx);
}

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH)
    {
        g_self = inst;
        DisableThreadLibraryCalls(inst);
        wchar_t p[MAX_PATH];
        GetModuleFileNameW(inst, p, MAX_PATH);
        g_dir = p;
        g_dir.resize(g_dir.find_last_of(L"\\/") + 1);
        _wfopen_s(&g_log, (g_dir + L"framegen.log").c_str(), L"w");
        LoadConfig();
        Log("framegen loaded (enabled=%d, forceSync=%d, motionScale=%d)",
            (int)g_enabled, g_forceSync, g_motionScale);

        // Install hooks from a worker thread (not allowed to do real work under loader lock).
        HANDLE t = CreateThread(nullptr, 0, HookThread, nullptr, 0, nullptr);
        if (t) CloseHandle(t);
    }
    return TRUE;
}
