#include "engine/crash_handler.h"
#include "engine/log.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <exception>

#if defined(_WIN32)
#  define NOMINMAX
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <DbgHelp.h>
#  pragma comment(lib, "Dbghelp.lib")
#endif

namespace qlike::crash {

namespace {

std::atomic<bool> g_installed{false};

#if defined(_WIN32)

void log_line(const char* line) {
    qlike::log::error(line);
    std::fprintf(stderr, "%s\n", line);
}

// Walk the given thread CONTEXT and log up to `max_frames` resolved frames.
// Uses DbgHelp's StackWalk64 — works without /Zi PDBs (we still get module +
// rva), much better with PDBs (file:line + symbol name).
void dump_stack_from_context(CONTEXT* ctx, int max_frames = 32) {
    HANDLE process = GetCurrentProcess();
    HANDLE thread  = GetCurrentThread();

    // SymInitialize is idempotent-ish — calling twice without SymCleanup is
    // tolerated. We don't SymCleanup because the process is dying anyway.
    static std::atomic<bool> sym_initialized{false};
    bool expected = false;
    if (sym_initialized.compare_exchange_strong(expected, true)) {
        SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_DEFERRED_LOADS |
                      SYMOPT_UNDNAME);
        SymInitialize(process, nullptr, TRUE);
    }

    STACKFRAME64 frame{};
    DWORD machine = 0;
#if defined(_M_X64)
    machine = IMAGE_FILE_MACHINE_AMD64;
    frame.AddrPC.Offset    = ctx->Rip;
    frame.AddrFrame.Offset = ctx->Rbp;
    frame.AddrStack.Offset = ctx->Rsp;
#elif defined(_M_IX86)
    machine = IMAGE_FILE_MACHINE_I386;
    frame.AddrPC.Offset    = ctx->Eip;
    frame.AddrFrame.Offset = ctx->Ebp;
    frame.AddrStack.Offset = ctx->Esp;
#elif defined(_M_ARM64)
    machine = IMAGE_FILE_MACHINE_ARM64;
    frame.AddrPC.Offset    = ctx->Pc;
    frame.AddrFrame.Offset = ctx->Fp;
    frame.AddrStack.Offset = ctx->Sp;
#else
    log_line("[crash] unsupported arch for stack walk");
    return;
#endif
    frame.AddrPC.Mode    = AddrModeFlat;
    frame.AddrFrame.Mode = AddrModeFlat;
    frame.AddrStack.Mode = AddrModeFlat;

    char sym_buf[sizeof(SYMBOL_INFO) + 512];
    auto* sym = reinterpret_cast<SYMBOL_INFO*>(sym_buf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen   = 511;

    log_line("[crash] --- begin stack trace ---");
    for (int i = 0; i < max_frames; ++i) {
        BOOL ok = StackWalk64(machine, process, thread, &frame, ctx,
                              nullptr, SymFunctionTableAccess64,
                              SymGetModuleBase64, nullptr);
        if (!ok || frame.AddrPC.Offset == 0) break;

        DWORD64 addr = frame.AddrPC.Offset;
        DWORD64 displacement = 0;
        const char* name = "?";
        if (SymFromAddr(process, addr, &displacement, sym)) {
            name = sym->Name;
        }

        IMAGEHLP_LINE64 line{};
        line.SizeOfStruct = sizeof(line);
        DWORD line_disp = 0;
        char buf[1024];
        if (SymGetLineFromAddr64(process, addr, &line_disp, &line)) {
            std::snprintf(buf, sizeof(buf),
                          "[crash] #%2d  0x%016llX  %s  (%s:%lu +%lu)",
                          i, static_cast<unsigned long long>(addr),
                          name, line.FileName,
                          static_cast<unsigned long>(line.LineNumber),
                          static_cast<unsigned long>(line_disp));
        } else {
            std::snprintf(buf, sizeof(buf),
                          "[crash] #%2d  0x%016llX  %s +0x%llX",
                          i, static_cast<unsigned long long>(addr),
                          name, static_cast<unsigned long long>(displacement));
        }
        log_line(buf);
    }
    log_line("[crash] --- end stack trace ---");
}

const char* seh_code_name(DWORD code) {
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION:         return "ACCESS_VIOLATION";
        case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:    return "ARRAY_BOUNDS_EXCEEDED";
        case EXCEPTION_BREAKPOINT:               return "BREAKPOINT";
        case EXCEPTION_DATATYPE_MISALIGNMENT:    return "DATATYPE_MISALIGNMENT";
        case EXCEPTION_FLT_DENORMAL_OPERAND:     return "FLT_DENORMAL_OPERAND";
        case EXCEPTION_FLT_DIVIDE_BY_ZERO:       return "FLT_DIVIDE_BY_ZERO";
        case EXCEPTION_FLT_INEXACT_RESULT:       return "FLT_INEXACT_RESULT";
        case EXCEPTION_FLT_INVALID_OPERATION:    return "FLT_INVALID_OPERATION";
        case EXCEPTION_FLT_OVERFLOW:             return "FLT_OVERFLOW";
        case EXCEPTION_FLT_STACK_CHECK:          return "FLT_STACK_CHECK";
        case EXCEPTION_FLT_UNDERFLOW:            return "FLT_UNDERFLOW";
        case EXCEPTION_ILLEGAL_INSTRUCTION:      return "ILLEGAL_INSTRUCTION";
        case EXCEPTION_IN_PAGE_ERROR:            return "IN_PAGE_ERROR";
        case EXCEPTION_INT_DIVIDE_BY_ZERO:       return "INT_DIVIDE_BY_ZERO";
        case EXCEPTION_INT_OVERFLOW:             return "INT_OVERFLOW";
        case EXCEPTION_INVALID_DISPOSITION:      return "INVALID_DISPOSITION";
        case EXCEPTION_NONCONTINUABLE_EXCEPTION: return "NONCONTINUABLE_EXCEPTION";
        case EXCEPTION_PRIV_INSTRUCTION:         return "PRIV_INSTRUCTION";
        case EXCEPTION_SINGLE_STEP:              return "SINGLE_STEP";
        case EXCEPTION_STACK_OVERFLOW:           return "STACK_OVERFLOW";
        default:                                  return "UNKNOWN";
    }
}

LONG WINAPI seh_filter(EXCEPTION_POINTERS* ep) {
    if (!ep || !ep->ExceptionRecord || !ep->ContextRecord) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "[crash] SEH 0x%08lX (%s) at 0x%016llX",
                  static_cast<unsigned long>(code),
                  seh_code_name(code),
                  reinterpret_cast<unsigned long long>(
                      ep->ExceptionRecord->ExceptionAddress));
    log_line(buf);

    // For AV, the first parameter is the operation (0=read, 1=write, 8=DEP),
    // second is the offending address.
    if (code == EXCEPTION_ACCESS_VIOLATION &&
        ep->ExceptionRecord->NumberParameters >= 2) {
        const ULONG_PTR op   = ep->ExceptionRecord->ExceptionInformation[0];
        const ULONG_PTR addr = ep->ExceptionRecord->ExceptionInformation[1];
        const char* opname = (op == 0 ? "read" :
                              op == 1 ? "write" :
                              op == 8 ? "DEP"   : "?");
        std::snprintf(buf, sizeof(buf),
                      "[crash] AV: %s of 0x%016llX",
                      opname, static_cast<unsigned long long>(addr));
        log_line(buf);
    }

    dump_stack_from_context(ep->ContextRecord);
    qlike::log::close();    // flush
    return EXCEPTION_EXECUTE_HANDLER;   // process will die after we return
}

void terminate_handler() {
    log_line("[crash] std::terminate called");
    if (auto p = std::current_exception()) {
        try {
            std::rethrow_exception(p);
        } catch (const std::exception& e) {
            char buf[1024];
            std::snprintf(buf, sizeof(buf),
                          "[crash] terminate via std::exception: %s",
                          e.what() ? e.what() : "(null)");
            log_line(buf);
        } catch (...) {
            log_line("[crash] terminate via non-std exception");
        }
    } else {
        log_line("[crash] terminate without active exception");
    }
    // Stack at this point is the terminate site, not the throw site —
    // still useful as a coarse pointer.
    CONTEXT ctx{};
    RtlCaptureContext(&ctx);
    dump_stack_from_context(&ctx);
    qlike::log::close();
    std::abort();
}

#endif  // _WIN32

} // namespace

void install() {
#if defined(_WIN32)
    bool expected = false;
    if (!g_installed.compare_exchange_strong(expected, true)) return;
    SetUnhandledExceptionFilter(seh_filter);
    std::set_terminate(terminate_handler);
    qlike::log::info("[crash] handler installed (SEH + set_terminate)");
#endif
}

} // namespace qlike::crash
