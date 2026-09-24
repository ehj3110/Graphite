# SC Conformal — Surface Node Relaxation Ideas

> **Doc role:** Ideas / partial V4 lab. Agents: [AGENTS.md](../AGENTS.md) → [graphite/explicit/README.md](../graphite/explicit/README.md); implement only if tasked with V4 or a listed idea.

Status: **partially implemented in experimental V4** — see
`SC_CONFORMAL_COMPARE_V4_SURFACE_RELAX.md` and
`graphite/explicit/compare_v4_surface_relax/`. Ideas 1+2 MVP: tangent Laplacian
with sharp CAD features pinned (crease *sliding* not yet). Companion doc:
`SC_CONFORMAL_FUTURE_WORK.md`.

## Problem

After the closest-point morph (plus the stair-step normal gate added Aug 2026),
iron nodes are frozen wherever projection dropped them. The Laplacian relax in
`morph_hex_scaffold` only moves *interior* nodes. Consequences:

- A gated corner that had to travel far in one axis (e.g. the notch corner at
  (96, 132, 56) on the 6×6×4 wrist rest, redirected 4.4 mm in +Y) lands on the
  surface but crowds its neighbours; the surrounding skin looks stretched or
  pinched.
- More generally, surface node spacing inherits every artifact of the
  projection step: clamped nodes, residual snaps, and redirects all leave
  uneven skin edge lengths that nothing downstream corrects.

Goal: a relaxation pass that evens out surface node spacing **while keeping
every node on the CAD surface** and without re-introducing the failure we just
fixed (nodes sliding around a stair-step crease onto the wrong patch).

Anchor points in code:

- `graphite/explicit/conformal_core.py` — `morph_hex_scaffold` (projection +
  interior relax), `redirect_interior_pull_to_exposed_normals` (stair gate),
  `exposed_faces_with_outward_normals` (skin connectivity source).
- Skin edge graph = edges of exposed faces; surface neighbours come from
  there, not from the full hex edge graph.

---

## Idea 1 — Tangent-constrained Laplacian with back-projection (preferred)

Iterate a few passes over iron nodes only:

1. Compute the Laplacian target: average of the node's *surface* neighbours
   (exposed-face edge graph).
2. Project the displacement onto the local tangent plane (normal taken from
   the nearest CAD surface point) — tangent-only stepping avoids the classic
   Laplacian shrinkage.
3. Re-snap the moved node to the CAD with closest-point so it never leaves the
   surface.

Pros: simple, directly spreads out the crowding a redirected node creates.
Cons: near a crease, closest-point re-snap can flip a node onto the adjacent
patch — must be combined with Idea 2. Costs one nearest-point query per node
per iteration.

## Idea 2 — Patch/crease locking (required companion to any of these)

Classify surface nodes by CAD patch before relaxing:

- Cluster CAD normals at each node's location (floor, top, each wall family).
- Nodes strictly inside a patch: relax freely within that patch.
- Nodes on a crease (two patches): constrain motion to slide *along* the
  crease curve only.
- Nodes at corners (three+ patches): pin.

This is the standard feature-preserving smoothing recipe. Without it, Idea 1
can undo the stair-step normal gate by letting a wall node migrate onto the
floor. Detection: dihedral angle threshold on CAD face normals near the node,
or compare normals of the node's own exposed faces.

## Idea 3 — Edge-length equalization springs

Same tangent-plane + re-snap machinery as Idea 1, different displacement rule:
springs push every skin edge toward the median skin edge length. Converges
more directly to "evenly spaced" than Laplacian averaging (which equalizes
positions, not lengths, and can leave anisotropic spacing). Precedent in the
codebase: `apply_radial_equalization_relaxation` (sphere-only) uses the same
spirit.

## Idea 4 — Couple surface DOFs into the existing global relax

Instead of "project → freeze → relax interior", give iron nodes partial
freedom during the global Laplacian relax: fixed along the surface normal,
free in the tangent plane. Surface and volume equilibrate together, so an
interior node is never torn between a badly placed anchor and its neighbours.

Cleanest result in principle; touchiest in practice:

- Needs a nearest-point query per surface node per relax iteration (or a
  cached normal field refreshed every k iterations).
- All established quality/clamp guarantees (collapse warnings, inversion
  checks) must be re-verified, since anchors are no longer fixed.

## Idea 5 — Quality-driven smoothing on the exposed quad mesh

The exposed faces form a quad mesh. Optimize quad corner angles (Winslow or
angle-based smoothing à la Zhou–Shimada) restricted to the surface, instead of
raw positions. Best element shapes near concave corners where Laplacian tends
to flatten or fold. Most implementation effort — hold in reserve unless
Ideas 1–3 aren't good enough.

---

## Common pitfalls (apply to every idea)

- **Cap per-iteration travel** to a fraction of cell size so a node cannot
  migrate far from its home cell.
- **Keep the collapsed/inverted-hex checks on**: sliding the skin reshapes the
  bricks behind it.
- **Re-run (or interleave) the interior relax afterward** so the volume
  follows the smoothed skin.
- **Convergence guard**: stop when max tangential step < tolerance; don't run
  a fixed large iteration count blindly.

## Recommended order of attack

1. Idea 1 + Idea 2 together (tangent Laplacian with patch/crease locking) as
   an opt-in flag on `morph_hex_scaffold` (e.g. `surface_relax_iterations`).
2. If spacing is still uneven, swap the averaging rule for Idea 3's springs.
3. Ideas 4 and 5 only if the wrist-rest renders still show visible pinching.

Validation: reuse `scripts/diagnose_stair_step_gate.py` metrics (stranded
nodes, collapsed/inverted counts, min volume ratio) plus a new skin edge
length histogram (std/median before vs after) on both 6×6×4 and 12×12×4.
