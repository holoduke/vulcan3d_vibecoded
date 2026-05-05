#include "engine/log.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

namespace qlike::log {

namespace {
std::mutex g_mu;
std::ofstream g_file;

const char* level_str(int level) {
    switch (level) {
        case 0: return "INFO ";
        case 1: return "WARN ";
        case 2: return "ERROR";
        default: return "?????";
    }
}

void write_line(int level, std::string_view msg) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) % 1000;
    std::tm tm{};
    localtime_s(&tm, &t);
    char ts[32];
    std::snprintf(ts, sizeof(ts), "%02d:%02d:%02d.%03lld",
                  tm.tm_hour, tm.tm_min, tm.tm_sec,
                  static_cast<long long>(ms.count()));

    auto& sink = g_file.is_open() ? static_cast<std::ostream&>(g_file)
                                  : std::cerr;
    sink << ts << " " << level_str(level) << " " << msg << "\n";
    sink.flush();
    if (g_file.is_open()) {
        // Mirror to stdout/stderr too so the user sees something live.
        auto& mirror = (level >= 1) ? std::cerr : std::cout;
        mirror << ts << " " << level_str(level) << " " << msg << "\n";
        mirror.flush();
    }
}

std::string vformat(const char* fmt, std::va_list ap) {
    std::va_list ap2;
    va_copy(ap2, ap);
    int n = std::vsnprintf(nullptr, 0, fmt, ap2);
    va_end(ap2);
    if (n < 0) return {};
    std::vector<char> buf(static_cast<size_t>(n) + 1);
    std::vsnprintf(buf.data(), buf.size(), fmt, ap);
    return std::string(buf.data(), static_cast<size_t>(n));
}
} // namespace

void open(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_file.is_open()) g_file.close();
    g_file.open(path, std::ios::out | std::ios::trunc);
}

void close() {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_file.is_open()) g_file.close();
}

void info(std::string_view m)  { write_line(0, m); }
void warn(std::string_view m)  { write_line(1, m); }
void error(std::string_view m) { write_line(2, m); }

void infof(const char* fmt, ...)  { va_list ap; va_start(ap, fmt); info(vformat(fmt, ap));  va_end(ap); }
void warnf(const char* fmt, ...)  { va_list ap; va_start(ap, fmt); warn(vformat(fmt, ap));  va_end(ap); }
void errorf(const char* fmt, ...) { va_list ap; va_start(ap, fmt); error(vformat(fmt, ap)); va_end(ap); }

} // namespace qlike::log
