@echo off
rem Run from "x64 Native Tools Command Prompt for VS" (VS 2017 or newer + Windows SDK).
cl /nologo /LD /O2 /MT /EHsc /std:c++17 framegen.cpp /Fe:d3d11.dll /link /DEF:framegen.def
if errorlevel 1 goto fail
cl /nologo /LD /O2 /MT /EHsc /std:c++17 framegen.cpp /Fe:framegen.dll
if errorlevel 1 goto fail
cl /nologo /O2 /MT /EHsc inject.cpp /Fe:inject.exe
if errorlevel 1 goto fail
echo.
echo Built: d3d11.dll (proxy), framegen.dll (for inject.exe), inject.exe
goto end
:fail
echo BUILD FAILED
:end
