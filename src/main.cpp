#include "engine/crash_handler.h"
#include "engine/log.h"
#include "engine/vk_engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <typeinfo>

namespace {

struct Args {
    qlike::RunOptions opts;
    std::string log_path = "qlike.log";
};

bool parse_args(int argc, char** argv, Args& out) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after %s\n", what);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--frames") {
            const char* v = next("--frames"); if (!v) return false;
            out.opts.max_frames = std::atoi(v);
        } else if (a == "--screenshot") {
            const char* v = next("--screenshot"); if (!v) return false;
            out.opts.screenshot_path = v;
        } else if (a == "--screenshot-after") {
            const char* v = next("--screenshot-after"); if (!v) return false;
            out.opts.screenshot_after_frames = std::atoi(v);
        } else if (a == "--log") {
            const char* v = next("--log"); if (!v) return false;
            out.log_path = v;
        } else if (a == "--autodemo") {
            const char* v = next("--autodemo"); if (!v) return false;
            out.opts.autodemo_seconds = static_cast<float>(std::atof(v));
        } else if (a == "--help" || a == "-h") {
            std::printf("usage: quake_like [options]\n"
                        "  --frames N            Run N frames then exit (-1 = forever)\n"
                        "  --screenshot PATH     Capture one frame to PATH (PPM) and exit\n"
                        "  --screenshot-after N  Wait N frames before capturing (default 5)\n"
                        "  --log PATH            Log file path (default qlike.log)\n"
                        "  --autodemo SECS       Synthesise walk+fire input for SECS, then exit\n");
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Args args{};
    if (!parse_args(argc, argv, args)) return 2;

    qlike::log::open(args.log_path);
    qlike::crash::install();
    qlike::log::infof("quake-like startup (argc=%d)", argc);

    int rc = 0;
    try {
        qlike::VulkanEngine engine;
        engine.init();
        engine.run(args.opts);
        engine.shutdown();
    } catch (const std::exception& e) {
        // Log type via typeid (safe — rdata) and DELIBERATELY do not touch
        // e.what(). On the gameplay-stall path the runtime_error's
        // heap-allocated message is freed before we catch (we've reproduced
        // an AV here on `raw[scan_len]` read). The throw site already logs
        // the cause via QLIKE_VK_CHECK / vk_check_loc, so the message is
        // already in the log file — the catch only needs to record that we
        // exited via exception.
        const char* type = typeid(e).name();
        qlike::log::errorf("fatal: caught std::exception (type=%s) — see "
                           "earlier '[vk_check]' line for cause", type);
        std::fprintf(stderr, "fatal: caught std::exception (type=%s)\n", type);
        rc = 1;
    } catch (...) {
        qlike::log::error("fatal: caught non-std exception (type unknown)");
        std::fprintf(stderr, "fatal: caught non-std exception (type unknown)\n");
        rc = 1;
    }

    qlike::log::info("quake-like shutdown");
    qlike::log::close();
    return rc;
}
