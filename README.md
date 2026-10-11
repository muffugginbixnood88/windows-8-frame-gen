# framegen - GPU-side 2x frame generation for windows8.1 64-bit DX11 games (vsync-paced)

Build: run build.bat from an "x64 Native Tools Command Prompt for VS".
  -> d3d11.dll (proxy), framegen.dll (injectable copy), inject.exe

## Install - pick one
A) Proxy (simplest): copy d3d11.dll + framegen.ini next to the game's 64-bit .exe.
   Remove by deleting d3d11.dll.
B) Inject (for games that don't pick up a local d3d11.dll): start the game, then run
   inject.exe <game.exe> framegen.dll      (framegen.ini must sit next to framegen.dll)
   
v1.0- press F10.
V1.1- 10 second delay.
## Pacing (capped base frames)
Every real frame is followed by an interpolated one, both presented with vsync, so
output = monitor refresh and the game's base fps is capped at half of it
(60 Hz -> 30 base / 60 out, 144 Hz -> 72 base / 144 out). Turn the game's own vsync on.

## Works with
64-bit Direct3D 11 games (windowed, borderless or exclusive fullscreen), including
sRGB / BGRA / typeless / 10-bit / FP16 backbuffers. Launcher and UI swapchains are ignored.

## Does not work with
DX9 / DX10 / DX12 / Vulkan / OpenGL games, 32-bit games, MSAA backbuffers,
Microsoft Store / UWP games.

## Warning
Proxy DLLs and injection are flagged by anti-cheat (EAC, BattlEye, Vanguard, VAC on
protected servers...). Use only in single-player / offline games.
If a game crashes at start, delete d3d11.dll and read framegen.log.
