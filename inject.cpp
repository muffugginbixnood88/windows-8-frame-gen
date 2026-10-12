// inject.cpp - minimal LoadLibrary injector (64-bit).
// Usage: inject.exe <pid | process-name.exe> <path\to\framegen.dll>
// Use it for games that don't load a proxy d3d11.dll from their folder.
#include <windows.h>
#include <tlhelp32.h>
#include <cstdio>
#include <cstdlib>
#include <cwchar>

static DWORD FindPid(const wchar_t* name)
{
    PROCESSENTRY32W pe{}; pe.dwSize = sizeof(pe);
    HANDLE s = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    DWORD pid = 0;
    if (s != INVALID_HANDLE_VALUE && Process32FirstW(s, &pe))
        do { if (!_wcsicmp(pe.szExeFile, name)) { pid = pe.th32ProcessID; break; } }
        while (Process32NextW(s, &pe));
    if (s != INVALID_HANDLE_VALUE) CloseHandle(s);
    return pid;
}

int wmain(int argc, wchar_t** argv)
{
    if (argc < 3) { wprintf(L"usage: inject <pid | name.exe> <framegen.dll>\n"); return 1; }
    DWORD pid = wcstoul(argv[1], nullptr, 10);
    if (!pid) pid = FindPid(argv[1]);
    if (!pid) { wprintf(L"process not found\n"); return 1; }

    wchar_t full[MAX_PATH];
    GetFullPathNameW(argv[2], MAX_PATH, full, nullptr);
    if (GetFileAttributesW(full) == INVALID_FILE_ATTRIBUTES) { wprintf(L"dll not found: %s\n", full); return 1; }

    HANDLE h = OpenProcess(PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION |
                           PROCESS_VM_WRITE | PROCESS_VM_READ, FALSE, pid);
    if (!h) { wprintf(L"OpenProcess failed (%lu) - try an elevated prompt\n", GetLastError()); return 1; }

    SIZE_T bytes = (wcslen(full) + 1) * sizeof(wchar_t);
    void* remote = VirtualAllocEx(h, nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!remote || !WriteProcessMemory(h, remote, full, bytes, nullptr)) { wprintf(L"write failed\n"); return 1; }

    auto ll = (LPTHREAD_START_ROUTINE)GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "LoadLibraryW");
    HANDLE t = CreateRemoteThread(h, nullptr, 0, ll, remote, 0, nullptr);
    if (!t) { wprintf(L"CreateRemoteThread failed (%lu)\n", GetLastError()); return 1; }
    WaitForSingleObject(t, 10000);
    DWORD ec = 0; GetExitCodeThread(t, &ec);
    VirtualFreeEx(h, remote, 0, MEM_RELEASE);
    CloseHandle(t); CloseHandle(h);
    wprintf(ec ? L"injected into pid %lu\n" : L"LoadLibrary returned 0 (wrong architecture or blocked)\n", pid);
    return ec ? 0 : 1;
}
