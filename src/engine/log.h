#pragma once

#include <string>
#include <string_view>

namespace qlike::log {

// Open the log file. Subsequent writes append to it. Idempotent — calling open
// again replaces the previous file.
void open(const std::string& path);

void close();

// Write a line with a timestamp + level prefix. Flushes after every call so
// crashes don't lose the last message.
void info(std::string_view msg);
void warn(std::string_view msg);
void error(std::string_view msg);

// Printf-style helpers (use sparingly — the variadic forms allocate).
void infof(const char* fmt, ...);
void warnf(const char* fmt, ...);
void errorf(const char* fmt, ...);

} // namespace qlike::log
