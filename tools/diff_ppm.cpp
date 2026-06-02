// Pixel-wise PPM diff. Reports mean abs diff %, max single-pixel diff,
// and a count of pixels above the 8% delta threshold. Exit code 0 if
// mean SAD < 1.5%, 1 otherwise — so CI / migration scripts can gate on
// the threshold directly.
//
// Build: cl /EHsc /O2 tools/diff_ppm.cpp /Fe:build/diff_ppm.exe
// Usage: diff_ppm reference.ppm current.ppm

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct PPM {
    int w = 0, h = 0;
    std::vector<uint8_t> px;     // RGB triplets
};

static bool read_ppm(const char* path, PPM& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "open %s: failed\n", path); return false; }
    char magic[3] = {};
    if (std::fread(magic, 1, 2, f) != 2 || magic[0] != 'P' || magic[1] != '6') {
        std::fprintf(stderr, "%s: not a binary PPM (P6)\n", path);
        std::fclose(f); return false;
    }
    auto skip_ws_and_comments = [&]() {
        int c;
        for (;;) {
            c = std::fgetc(f);
            if (c == EOF) return;
            if (c == '#') {
                while (c != '\n' && c != EOF) c = std::fgetc(f);
                continue;
            }
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') continue;
            std::ungetc(c, f);
            return;
        }
    };
    int maxv = 0;
    skip_ws_and_comments();
    if (std::fscanf(f, "%d", &out.w) != 1) { std::fclose(f); return false; }
    skip_ws_and_comments();
    if (std::fscanf(f, "%d", &out.h) != 1) { std::fclose(f); return false; }
    skip_ws_and_comments();
    if (std::fscanf(f, "%d", &maxv) != 1) { std::fclose(f); return false; }
    std::fgetc(f);     // single whitespace after maxval
    if (maxv != 255) {
        std::fprintf(stderr, "%s: maxv %d != 255 (unsupported)\n", path, maxv);
        std::fclose(f); return false;
    }
    size_t bytes = static_cast<size_t>(out.w) * out.h * 3;
    out.px.resize(bytes);
    size_t got = std::fread(out.px.data(), 1, bytes, f);
    std::fclose(f);
    if (got != bytes) {
        std::fprintf(stderr, "%s: short read (%zu of %zu)\n", path, got, bytes);
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: diff_ppm <reference.ppm> <current.ppm>\n");
        return 2;
    }
    PPM a, b;
    if (!read_ppm(argv[1], a)) return 2;
    if (!read_ppm(argv[2], b)) return 2;
    if (a.w != b.w || a.h != b.h) {
        std::fprintf(stderr, "size mismatch: ref=%dx%d cur=%dx%d\n",
                     a.w, a.h, b.w, b.h);
        return 2;
    }
    const size_t n = a.px.size();
    uint64_t sad_sum = 0;
    int max_pix_delta = 0;
    size_t pix_above_8pct = 0;
    const size_t pixel_count = static_cast<size_t>(a.w) * a.h;
    for (size_t i = 0; i < pixel_count; ++i) {
        int dr = std::abs(int(a.px[i * 3 + 0]) - int(b.px[i * 3 + 0]));
        int dg = std::abs(int(a.px[i * 3 + 1]) - int(b.px[i * 3 + 1]));
        int db = std::abs(int(a.px[i * 3 + 2]) - int(b.px[i * 3 + 2]));
        int pix_sad = dr + dg + db;     // 0..765
        sad_sum += pix_sad;
        if (pix_sad > max_pix_delta) max_pix_delta = pix_sad;
        if (pix_sad > 61) ++pix_above_8pct;     // 8% of 765 ≈ 61
    }
    double mean_pct = (sad_sum / static_cast<double>(n)) * (100.0 / 255.0);
    double max_pct  = max_pix_delta * (100.0 / 765.0);
    double above_pct = (pix_above_8pct / static_cast<double>(pixel_count)) * 100.0;
    std::printf("size=%dx%d  mean=%.3f%%  max=%.2f%%  above8%%=%.3f%%  ",
                a.w, a.h, mean_pct, max_pct, above_pct);
    bool ok = (mean_pct < 1.5) && (above_pct < 0.5);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
