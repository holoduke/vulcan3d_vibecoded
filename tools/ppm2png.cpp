// Minimal P6 PPM -> PNG converter so terrain/material changes can be
// self-verified from the in-game F12 / --screenshot capture (which
// writes binary P6 PPM). Build: see tools/build_ppm2png.ps1.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static int skip_ws_comments(FILE* f) {
    int c;
    for (;;) {
        c = fgetc(f);
        if (c == '#') { while ((c = fgetc(f)) != '\n' && c != EOF) {} continue; }
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') continue;
        return c;
    }
}
static int read_uint(FILE* f) {
    int c = skip_ws_comments(f);
    int v = 0;
    while (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); c = fgetc(f); }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: ppm2png in.ppm out.png\n"); return 2; }
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "open fail: %s\n", argv[1]); return 1; }
    char magic[3] = {0};
    if (std::fread(magic, 1, 2, f) != 2 || std::strcmp(magic, "P6") != 0) {
        std::fprintf(stderr, "not P6 PPM\n"); std::fclose(f); return 1;
    }
    int w = read_uint(f), h = read_uint(f), maxv = read_uint(f);
    (void)maxv;
    if (w <= 0 || h <= 0) { std::fprintf(stderr, "bad dims\n"); std::fclose(f); return 1; }
    std::vector<unsigned char> px((size_t)w * h * 3);
    if (std::fread(px.data(), 1, px.size(), f) != px.size()) {
        std::fprintf(stderr, "short read\n"); std::fclose(f); return 1;
    }
    std::fclose(f);
    if (!stbi_write_png(argv[2], w, h, 3, px.data(), w * 3)) {
        std::fprintf(stderr, "png write fail\n"); return 1;
    }
    std::printf("%s -> %s (%dx%d)\n", argv[1], argv[2], w, h);
    return 0;
}
