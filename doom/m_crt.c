// SPDX-License-Identifier: GPL-2.0-or-later
// The CUDA CRT's dials, as DOOM's menu sees them. Compiled into Crispy Doom.
//
// The compositor pass re-reads `set` every frame, so writing that file IS the live
// preview. `tube-profile.set` is the same text kept for the next session. The slider
// panel (tools/crt-config) owns the same two files.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "m_crt.h"

typedef struct
{
    const char *name, *label, *unit;
    int page;
    float min, max, step, fallback, look;
    int decimals;
} crt_dial_t;

typedef struct
{
    const char *name, *values;
} crt_preset_t;

#include "m_crt_dials.h"

static float value[CRT_NUMDIALS];
static char extra[1024];   // pairs some other tool wrote: kept, never understood
static int preset_cursor = -1;

static const char *State(void)
{
    const char *dir = getenv("CUDA_CRT_STATE");
    return dir && *dir ? dir : NULL;
}

static int Path(char *out, size_t size, const char *name)
{
    const int n = snprintf(out, size, "%s/%s", State(), name);
    return n > 0 && (size_t) n < size;
}

int CRT_Available(void)
{
    return State() != NULL;
}

static int Find(const char *name)
{
    for (int i = 0; i < CRT_NUMDIALS; i++)
        if (!strcmp(crt_dials[i].name, name))
            return i;
    return -1;
}

static float Clamp(int i, float v)
{
    const crt_dial_t *d = &crt_dials[i];
    return v < d->min ? d->min : v > d->max ? d->max : v;
}

// Lay name=value pairs (commas or new lines) over the dials.
static void Apply(const char *text, int keep_unknown)
{
    char buf[4096], *save = NULL;

    snprintf(buf, sizeof(buf), "%s", text);
    for (char *p = strtok_r(buf, ",\n", &save); p; p = strtok_r(NULL, ",\n", &save))
    {
        char *eq = strchr(p, '='), *end = NULL;
        float v;
        int i;

        if (!eq)
            continue;
        *eq = '\0';
        p += strspn(p, " \t");
        p[strcspn(p, " \t")] = '\0';
        i = Find(p);
        v = strtof(eq + 1, &end);
        if (i >= 0)
        {
            if (end != eq + 1 && isfinite(v))
                value[i] = Clamp(i, v);
        }
        else if (keep_unknown && *p && strlen(extra) + strlen(p) + strlen(eq + 1) + 3 < sizeof(extra))
        {
            strcat(extra, p);
            strcat(extra, "=");
            strcat(extra, eq + 1);
            strcat(extra, ",");
        }
    }
}

static int ReadFile(const char *name, char *out, size_t size)
{
    char path[1024];
    FILE *f;
    size_t n;

    if (!State() || !Path(path, sizeof(path), name) || !(f = fopen(path, "r")))
        return 0;
    n = fread(out, 1, size - 1, f);
    out[n] = '\0';
    fclose(f);
    return 1;
}

void CRT_Load(void)
{
    char text[4096];

    extra[0] = '\0';
    for (int i = 0; i < CRT_NUMDIALS; i++)
        value[i] = crt_dials[i].fallback;   // what the pass uses for a dial nobody named
    if (ReadFile("set", text, sizeof(text)))
        Apply(text, 1);
}

// Whole file or nothing: the pass may read at any moment.
static void WriteFile(const char *name, const char *text)
{
    char path[1024], temp[1024], hidden[64];
    FILE *f;
    int ok;

    snprintf(hidden, sizeof(hidden), ".menu-%s", name);
    if (!State() || !Path(path, sizeof(path), name) || !Path(temp, sizeof(temp), hidden))
        return;
    if (!(f = fopen(temp, "w")))
    {
        fprintf(stderr, "CRT menu: cannot write %s\n", temp);
        return;
    }
    ok = fputs(text, f) >= 0;
    ok = fclose(f) == 0 && ok;
    if (!ok || rename(temp, path))
    {
        fprintf(stderr, "CRT menu: cannot save %s\n", path);
        unlink(temp);
    }
}

static void Save(void)
{
    char text[4096];
    size_t n = snprintf(text, sizeof(text), "%s", extra);

    for (int i = 0; i < CRT_NUMDIALS && n < sizeof(text); i++)
        n += snprintf(text + n, sizeof(text) - n, "%s=%.6g%s", crt_dials[i].name,
                      value[i], i + 1 < CRT_NUMDIALS ? "," : "\n");
    if (n >= sizeof(text))
        return;
    WriteFile("tube-profile.set", text);
    WriteFile("set", text);
}

int CRT_NumPages(void)
{
    return CRT_NUMPAGES;
}

const char *CRT_PageName(int page)
{
    return crt_pages[page];
}

int CRT_PageDials(int page)
{
    int n = 0;
    for (int i = 0; i < CRT_NUMDIALS; i++)
        n += crt_dials[i].page == page;
    return n;
}

static int Index(int page, int row)   // the table is sorted by page
{
    int first = 0;
    while (crt_dials[first].page != page)
        first++;
    return first + row;
}

const char *CRT_Label(int page, int row)
{
    return crt_dials[Index(page, row)].label;
}

void CRT_ValueText(int page, int row, char *out, int size)
{
    const int i = Index(page, row);
    const crt_dial_t *d = &crt_dials[i];
    snprintf(out, size, "%.*f%s%s", d->decimals, value[i], *d->unit ? " " : "", d->unit);
}

float CRT_Fraction(int page, int row)
{
    const int i = Index(page, row);
    return (value[i] - crt_dials[i].min) / (crt_dials[i].max - crt_dials[i].min);
}

void CRT_Nudge(int page, int row, int dir)
{
    const int i = Index(page, row);
    const crt_dial_t *d = &crt_dials[i];
    float steps;

    CRT_Load();
    // From a value between two stops, the first press lands on the next stop.
    steps = (value[i] - d->min) / d->step;
    steps = dir > 0 ? floorf(steps + 1.001f) : ceilf(steps - 1.001f);
    value[i] = Clamp(i, d->min + steps * d->step);
    Save();
}

int CRT_On(void)
{
    char path[1024];
    return State() && Path(path, sizeof(path), "off") && access(path, F_OK) != 0;
}

void CRT_Toggle(void)
{
    char path[1024];

    if (CRT_On())
        WriteFile("off", "");
    else if (State() && Path(path, sizeof(path), "off"))
        unlink(path);
}

static int Matches(const char *pairs)
{
    float before[CRT_NUMDIALS];
    int same = 1;

    memcpy(before, value, sizeof(value));
    Apply(pairs, 0);
    for (int i = 0; i < CRT_NUMDIALS; i++)
        same &= fabsf(before[i] - value[i]) <= 1e-4f * (1.0f + fabsf(value[i]));
    memcpy(value, before, sizeof(value));
    return same;
}

const char *CRT_PresetName(void)
{
    if (preset_cursor >= 0 && Matches(crt_presets[preset_cursor].values))
        return crt_presets[preset_cursor].name;
    for (int i = 0; i < CRT_NUMDIALS; i++)
        if (fabsf(value[i] - crt_dials[i].look) > 1e-4f * (1.0f + fabsf(value[i])))
            return "Custom";
    return "DOOM";
}

// A preset starts from the DOOM look, so what it shows does not depend on what came before.
static void Look(void)
{
    for (int i = 0; i < CRT_NUMDIALS; i++)
        value[i] = crt_dials[i].look;
}

void CRT_CyclePreset(int dir)
{
    CRT_Load();
    if (preset_cursor < 0)
        preset_cursor = dir > 0 ? 0 : CRT_NUMPRESETS - 1;
    else
        preset_cursor = (preset_cursor + (dir > 0 ? 1 : CRT_NUMPRESETS - 1)) % CRT_NUMPRESETS;
    Look();
    Apply(crt_presets[preset_cursor].values, 0);
    Save();
}

void CRT_Reset(void)
{
    CRT_Load();
    preset_cursor = -1;
    Look();
    Save();
}
