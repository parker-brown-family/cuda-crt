// Deterministic synthetic terminal content: dot-matrix "glyphs" in rows, a
// block cursor, and a scroll offset. No font, no files, same bytes every run.
#pragma once
#include <cstdint>
#include <vector>

inline uint32_t lcg(uint32_t &s) { s = s * 1664525u + 1013904223u; return s >> 8; }

// scroll: how many text rows the content has moved up. cursor_on: draw the cursor.
inline void draw_terminal(std::vector<uint8_t> &px, int w, int h, int scroll, bool cursor_on) {
    const int cw = 10, chh = 20, cols = w / cw, rows = h / chh;
    px.assign((size_t)w * h * 4, 255);
    for (size_t i = 0; i < px.size(); i += 4) { px[i] = 8; px[i + 1] = 12; px[i + 2] = 9; }
    for (int r = 0; r < rows; r++) {
        uint32_t s = (uint32_t)(r + scroll) * 2246822519u + 77u;
        int len = 8 + lcg(s) % (cols > 12 ? cols - 10 : 2);
        bool dim = lcg(s) % 5 == 0;
        for (int c = 1; c < len && c < cols - 1; c++) {
            uint32_t g = lcg(s);
            if (g % 7 == 0) continue;  // a space
            for (int gy = 0; gy < 7; gy++)
                for (int gx = 0; gx < 5; gx++) {
                    if (!((g >> ((gy * 5 + gx) % 23)) & 1)) continue;
                    for (int yy = 0; yy < 2; yy++)
                        for (int xx = 0; xx < 1 + (gx & 1); xx++) {
                            int X = c * cw + 1 + gx * 2 + xx - (gx & 1), Y = r * chh + 3 + gy * 2 + yy;
                            if (X < 0 || X >= w || Y >= h) continue;
                            uint8_t *p = &px[4 * ((size_t)Y * w + X)];
                            p[0] = dim ? 20 : 60; p[1] = dim ? 150 : 255; p[2] = dim ? 60 : 120;
                        }
                }
        }
        if (r == rows - 2 && cursor_on)
            for (int Y = r * chh + 2; Y < r * chh + chh - 2 && Y < h; Y++)
                for (int X = (len + 1) * cw; X < (len + 2) * cw && X < w; X++) {
                    uint8_t *p = &px[4 * ((size_t)Y * w + X)];
                    p[0] = 90; p[1] = 255; p[2] = 150;
                }
    }
}
