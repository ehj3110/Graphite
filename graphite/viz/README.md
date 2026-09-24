# `graphite.viz` — capability card

## Owns

Shared offscreen PyVista PNG framing helpers (plotter create/frame, fixed load arrows).

## Status

**Production** shared utilities.

## Public entrypoints

- `create_offscreen_plotter`, `frame_plotter_content`
- `add_fixed_load_arrow`, `add_fixed_load_arrows`, `subsample_load_face_sites`
- `viewport_world_height`

## Does not own

Aristo cross-section stress plots (`aristo.cross_section_viz`), Vocal flow plots (`lbm`), TO isosurface renders (`topt.topt_viz`).

## Mix-and-match

- Callers: Aristo / comparison scripts that need consistent PNG framing.

## Read next

1. Related UX notes: [docs/CROSS_SECTION_VIZ_HANDOFF.md](../../docs/CROSS_SECTION_VIZ_HANDOFF.md)
2. [graphite/aristo/README.md](../aristo/README.md)

## Do not open first

- Domain-specific plot modules in aristo/lbm/topt when only framing is needed — start here instead
