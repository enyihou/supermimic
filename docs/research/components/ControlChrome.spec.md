# ControlChrome Specification

## Overview

- **Target file:** `src/components/ControlChrome.tsx`
- **Screenshot:** `docs/design-references/official-demo-desktop-1440.png` and `official-demo-mobile-390.png`
- **Interaction model:** click-driven, keyboard-driven, and responsive sheet-driven

## Exact visual tokens

- Font families are `Inter Tight, Inter, Helvetica Neue, Arial, sans-serif` for display, `Inter, Helvetica Neue, Arial, sans-serif` for controls, and `JetBrains Mono, ui-monospace, Menlo, Consolas, monospace` for counters.
- `--page: #ffffff`, `--bg: #f5f5f7`, `--fill: #e8e8ed`, `--hair: #d2d2d7`, `--ink: #1d1d1f`, `--ink2: #6e6e73`.
- Glass uses `background: rgba(255,255,255,.72)`, dense glass uses `rgba(252,252,253,.88)`, `box-shadow: 0 0 0 .5px rgba(0,0,0,.08), 0 10px 30px rgba(0,0,0,.08), 0 1px 3px rgba(0,0,0,.06)`, and `backdrop-filter: saturate(180%) blur(24px)`.

## Desktop layout

- Legend top/left 20px. Heading is 17px, 600 weight, 1.2 line-height, margin-left 4px and 8px bottom.
- Segmented legend uses 4px padding, 999px radius, 2px gap. Segments are 30px high with 12px horizontal padding.
- SuperMimic pill top/right 20px, 36px high, 999px radius, 14px right and 11px left padding.
- Dock is 52px high, max-width calc viewport minus 40px, centered at bottom 20px, 6px horizontal padding, 4px gap, 999px radius.
- Dock icon buttons are 40px circles. Motion button is 260px wide, 40px high, 14px right and 12px left padding.

## Mobile layout

- At max-width 900px, dock height is 48px and controls become a fixed sheet above it. Motions button is 170px wide.
- At max-width 600px, edge inset is 12px and legend controls use 28px height with compact horizontal padding.
- Mobile screenshot shows a 331px-wide dock at x=29.5px for a 390px viewport.

## States

- Active icon or selector uses `#1d1d1f` background and white foreground. Hover uses rgba black 0.05. Active press uses rgba black 0.08.
- Policy switch is 40×24px with a 20px white knob. On state is dark track with knob translated 16px.
- Action-noise range is 0–0.25 and readout uses two decimals. Friction is 0–100% and disabled until boot completes.
- Motions sheet uses rise animation 0.28s `cubic-bezier(.28,.11,.32,1)`, 300px desktop width, max-height `min(60dvh,440px)`, 8px padding and 22px radius.
- Folder headers are 36px high, items 32px high, 10px horizontal padding, 10px radius, and active item uses dark background with white text.

## Text content

`Live imagination`, `World model`, `Goal`, `MuJoCo`, `step 0 since the last reset`, `Policy`, `on`, `Action noise`, `0.00`, `Friction`, `–`, `Motions`, `Filter motions`, and `© 2026 SuperMimic · License`.
