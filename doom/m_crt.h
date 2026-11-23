// SPDX-License-Identifier: GPL-2.0-or-later
// The CUDA CRT's dials, as DOOM's menu sees them. Compiled into Crispy Doom.
#ifndef M_CRT_H
#define M_CRT_H

// True when the CRT launcher started this game (it names its folder in CUDA_CRT_STATE).
int CRT_Available(void);
// Re-read the saved look. Every change re-reads first, so the slider panel can be open too.
void CRT_Load(void);

int CRT_NumPages(void);
const char *CRT_PageName(int page);
int CRT_PageDials(int page);
// Rows count from 0 within one page.
const char *CRT_Label(int page, int row);
void CRT_ValueText(int page, int row, char *out, int size);
float CRT_Fraction(int page, int row);
void CRT_Nudge(int page, int row, int dir);

int CRT_On(void);
void CRT_Toggle(void);
const char *CRT_PresetName(void);
void CRT_CyclePreset(int dir);
void CRT_Reset(void);

#endif
