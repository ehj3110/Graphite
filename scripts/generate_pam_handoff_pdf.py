#!/usr/bin/env python
"""
Generate Publication-Grade 5-Page PDF Handoff Report for Graphite Interlinked PAM Engine.

Target Audience: Simulation Engineers, Computational Mechanics Specialists, and MBD Researchers.
Focus: PAM lattice architecture, catenation logic, clearance scaling, grading, transitions,
       the 75x75x20mm flagship benchmark, AM supports, and simulation guidelines.
Excludes: Node/truss connection discussions (3.1/3.2) and boundary frames/mechanics (5).
"""
import base64
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUTPUTS_DIR = _REPO_ROOT / "outputs"
_ARTIFACT_DIR = Path(r"C:\Users\ehunt\.gemini\antigravity\brain\c3d5d97b-2716-45e7-97c0-6173eedf89c0")


def img_to_b64(path: Path) -> str:
    """Read image and return data URI."""
    if not path.exists():
        print(f"Warning: Image not found: {path}")
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = path.suffix.lower().replace(".", "")
    mime = "image/png" if ext == "png" else f"image/{ext}"
    return f"data:{mime};base64,{data}"


def build_html() -> str:
    # Encode images
    img_75 = img_to_b64(_OUTPUTS_DIR / "pam_transition_75x75x20_c6_to_d4.png")
    img_c6 = img_to_b64(_OUTPUTS_DIR / "c6tt_strip_75x25x20.png")
    img_d4 = img_to_b64(_OUTPUTS_DIR / "d4tet_strip_75x25x20.png")
    img_cube = img_to_b64(_OUTPUTS_DIR / "c6tt_3x3x3_12.7mm_d1.25.png")
    img_graded = img_to_b64(_OUTPUTS_DIR / "graded_pam_c6tt_thickness_gradient.png")
    img_morph = img_to_b64(_OUTPUTS_DIR / "transition_pam_c6_to_d4.png")
    img_supp = img_to_b64(_OUTPUTS_DIR / "trame_app_interlinked_supports.png")
    img_studio = img_to_b64(_OUTPUTS_DIR / "trame_app_interlinked_studio.png")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Graphite Interlinked Metamaterials: Lattice Architecture & Simulation Handoff Report</title>
<style>
  @page {{
    size: letter;
    margin: 11mm 13mm 11mm 13mm;
  }}
  *, *:before, *:after {{
    box-sizing: border-box;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b;
    background: #ffffff;
    line-height: 1.38;
    font-size: 8.8pt;
    margin: 0;
    padding: 0;
  }}
  h1, h2, h3, h4 {{
    color: #0f172a;
    font-weight: 700;
    margin-top: 0;
    line-height: 1.2;
  }}
  h1 {{
    font-size: 17.5pt;
    margin-bottom: 2px;
    color: #0f172a;
  }}
  .subtitle {{
    font-size: 10pt;
    color: #0284c7;
    font-weight: 600;
    margin-bottom: 10px;
  }}
  .meta-bar {{
    display: flex;
    justify-content: space-between;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    padding: 5px 10px;
    font-size: 7.8pt;
    color: #64748b;
    margin-bottom: 10px;
  }}
  .meta-bar span strong {{
    color: #0f172a;
  }}
  h2 {{
    font-size: 11.5pt;
    border-bottom: 1.5px solid #0284c7;
    padding-bottom: 2px;
    margin-top: 10px;
    margin-bottom: 6px;
  }}
  h3 {{
    font-size: 9.4pt;
    margin-top: 8px;
    margin-bottom: 3px;
    color: #1e293b;
  }}
  p {{
    margin: 0 0 6px 0;
    text-align: justify;
  }}
  ul, ol {{
    margin: 0 0 6px 0;
    padding-left: 18px;
  }}
  li {{
    margin-bottom: 3px;
  }}
  .logic-box {{
    background: #f0fdf4;
    border-left: 3.5px solid #16a34a;
    border-radius: 0 5px 5px 0;
    padding: 6px 9px;
    margin: 5px 0 8px 0;
    font-size: 8.4pt;
    line-height: 1.34;
  }}
  .logic-box .title {{
    font-weight: 700;
    color: #15803d;
    text-transform: uppercase;
    font-size: 7.2pt;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
  }}
  .sim-box {{
    background: #eff6ff;
    border-left: 3.5px solid #2563eb;
    border-radius: 0 5px 5px 0;
    padding: 6px 9px;
    margin: 5px 0 8px 0;
    font-size: 8.4pt;
    line-height: 1.34;
  }}
  .sim-box .title {{
    font-weight: 700;
    color: #1d4ed8;
    text-transform: uppercase;
    font-size: 7.2pt;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
  }}
  .figure-container {{
    text-align: center;
    margin: 6px 0;
    page-break-inside: avoid;
  }}
  .figure-container img {{
    max-width: 100%;
    height: auto;
    border-radius: 5px;
    border: 1px solid #cbd5e1;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }}
  .figure-caption {{
    font-size: 7.5pt;
    color: #64748b;
    margin-top: 3px;
    font-style: italic;
  }}
  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 6px;
    page-break-inside: avoid;
  }}
  .grid-3 {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 7px;
    margin-bottom: 6px;
    page-break-inside: avoid;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 7.8pt;
    margin: 6px 0 8px 0;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 4px 6px;
    text-align: left;
  }}
  th {{
    background: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
  }}
  tr:nth-child(even) td {{
    background: #f8fafc;
  }}
  .highlight {{
    color: #0284c7;
    font-weight: 700;
  }}
  .page-break {{
    page-break-before: always;
  }}
  .no-break {{
    page-break-inside: avoid;
  }}
  code {{
    font-family: Consolas, "Courier New", monospace;
    font-size: 8pt;
    background: #f1f5f9;
    padding: 1px 3px;
    border-radius: 3px;
    color: #0f172a;
  }}
  .formula-card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 4px 8px;
    margin: 4px 0;
    font-family: Consolas, monospace;
    font-size: 8.2pt;
    text-align: center;
    color: #0f172a;
  }}
</style>
</head>
<body>

<!-- ================= PAGE 1 ================= -->
<div class="page-container">
  <h1>Graphite Interlinked Metamaterials (PAMs)</h1>
  <div class="subtitle">Lattice Architecture, Catenation Logic, Clearance Physics &amp; Simulation Handoff Report</div>
  
  <div class="meta-bar">
    <span><strong>Target Audience:</strong> CAE Simulation Engineers &amp; Multi-Body Dynamics Researchers</span>
    <span><strong>Subsystem:</strong> <code>graphite.explicit.interlinked</code></span>
    <span><strong>Status:</strong> Validated Production Engine (100+ Tests Passing)</span>
  </div>

  <div class="logic-box">
    <div class="title">High-Level Logic Summary: What is a PAM in Plain English?</div>
    A standard 3D printed lattice is a <em>welded truss</em> where every strut is fused at common vertices. A <strong>Polycatenated Architected Material (PAM)</strong> is fundamentally different: it is true <strong>3D chainmail</strong> made of hundreds to thousands of <strong>independent, unbonded, hollow polyhedral wireframe cages</strong> that are topologically intertwined in space. Every cage floats with a strictly positive physical air gap (&Delta; &gt; 0) to its neighbors. Because there are zero solid welds between cages, the material exhibits a distinct <strong>kinematic free-play regime</strong>, followed by <strong>geometric contact locking</strong> and <strong>giant energy dissipation via sliding friction</strong>.
  </div>

  <h2>1. Foundational Mechanics: Why PAMs Behave Differently than Welded Lattices</h2>
  <p>
    When simulating standard explicit lattices (e.g. Kelvin, Octet, Kagome), finite element solvers rely on linear elastic continuum assumptions: struts stretch and bend axially, and deformation is governed by parent material Young's modulus and strut cross-sectional inertia. In PAMs, continuum assumptions fail because global compliance is dominated by <strong>contact mechanics and rigid-body free-play</strong>:
  </p>
  <ul>
    <li><strong>Phase I: Kinematic Articulation (Zero Initial Stiffness):</strong> Under small applied loads, cages slide and rotate freely within the clearance envelope without any material strain.</li>
    <li><strong>Phase II: Geometric Contact Jamming:</strong> As displacement grows, adjacent strut surfaces make contact across multiple coordination directions, forming an interconnected load-bearing network.</li>
    <li><strong>Phase III: Multi-Body Coulomb Friction Dissipation:</strong> High normal contact forces under cyclic or impact loading cause strut surfaces to slide against one another. Inter-cage friction (&mu; &asymp; 0.15 &ndash; 0.35) dissipates massive kinetic and acoustic energy without plastic fatigue of the base metal or polymer.</li>
  </ul>

  <h2>2. How We Construct 3D Interlocking Lattices</h2>

  <div class="logic-box">
    <div class="title">High-Level Logic Summary: How do cages interlink in 3D without colliding?</div>
    In 2D, linking chainmail rings is easy. But in 3D, if you place closed cages at cubic lattice sites, how do you prevent them from crashing into each other along all 3 axes at once? Simple shapes like cuboctahedra fail because 3D rotations do not commute&mdash;you cannot twist a cage by 45&deg; in X, Y, and Z simultaneously. Our solution uses <strong>Archimedean Truncated Tetrahedra (C-6-TT)</strong>, which have open hexagonal and triangular window cutouts naturally facing all 6 Cartesian neighbor directions. Adjacent cages slide right through these open windows with <em>zero relative twist</em> and zero collisions. On diamond networks, we use <strong>Regular Tetrahedra (D-4-TET)</strong> on a bipartite grid with a 60&deg; bond twist, achieving 4-fold corner-through-face catenation.
  </div>

  <p>
    The fundamental mathematical challenge in 3D catenation is avoiding <strong>rotational frustration</strong>:
  </p>
  <ul>
    <li><strong>The Cuboctahedron Paradox:</strong> Setting a standard wireframe cuboctahedron on a simple cubic (<code>pcu</code>) grid causes collisions (&Delta; = 0.0 mm). Because orthogonal SO(3) rotations do not commute ([R<sub>x</sub>, R<sub>y</sub>] &ne; 0), no static 3D rotation matrix can align face windows for clearance along all three Cartesian axes at once.</li>
    <li><strong>The C-6-TT Breakthrough (Zhou et al., <em>Science</em> 2025):</strong> Truncating a regular tetrahedron at 1/3 edge length yields 12 vertices and 18 struts. Its 4 hexagonal faces align with &lang;1, 1, 1&rang; directions, and its 4 triangular corner windows align with Cartesian approach vectors. Neighboring cages penetrate through these windows unrotated, achieving <strong>6-fold coordination (z = 6)</strong> with generous positive clearance.</li>
  </ul>
</div>

<!-- ================= PAGE 2 ================= -->
<div class="page-break"></div>
<div class="page-container">
  <ul>
    <li><strong>The D-4-TET Diamond Architecture:</strong> On diamond-cubic networks (<code>dia</code>, coordination z = 4), cages are regular tetrahedra (4 vertices, 6 struts). The engine splits the lattice into bipartite Sublattices A and B. Sublattice B is inverted and twisted by 60&deg; relative to Sublattice A along the tetrahedral bond axis, enabling <strong>corner-through-face catenation</strong>.</li>
    <li><strong>Topological Proof (Gauss Linking Integral):</strong> To verify true multi-body catenation prior to meshing, the engine computes the double Gauss linking path integral:
      <div class="formula-card">Lk(&gamma;<sub>1</sub>, &gamma;<sub>2</sub>) = (1 / 4&pi;) &oint; &oint; [ (r<sub>1</sub> &minus; r<sub>2</sub>) &middot; (dr<sub>1</sub> &times; dr<sub>2</sub>) ] / |r<sub>1</sub> &minus; r<sub>2</sub>|<sup>3</sup> &ne; 0</div>
      A non-zero linking number mathematically guarantees that cages cannot be separated without cutting.
    </li>
  </ul>

  <div class="grid-3" style="margin-top: 4px; margin-bottom: 4px;">
    <div class="figure-container" style="margin: 2px 0;">
      <img src="{img_c6}" alt="C6-TT Strip" style="max-height: 102px;">
      <div class="figure-caption"><strong>Figure 1:</strong> C-6-TT strip (75 &times; 25 &times; 20 mm, 300 cages, &Delta; = 157 &mu;m).</div>
    </div>
    <div class="figure-container" style="margin: 2px 0;">
      <img src="{img_d4}" alt="D4-TET Strip" style="max-height: 102px;">
      <div class="figure-caption"><strong>Figure 2:</strong> D-4-TET strip (80.8 &times; 23.1 &times; 23.1 mm, 112 cages, &Delta; = 427 &mu;m).</div>
    </div>
    <div class="figure-container" style="margin: 2px 0;">
      <img src="{img_cube}" alt="Macroscopic Cube" style="max-height: 102px;">
      <div class="figure-caption"><strong>Figure 3:</strong> 3 &times; 3 &times; 3 C-6-TT cube (a<sub>0</sub> = 12.7 mm, &Delta; = 418 &mu;m).</div>
    </div>
  </div>

  <h2 style="margin-top: 6px;">3. The Lattice Generation Engine &amp; Clearance Scaling Physics</h2>

  <div class="logic-box" style="margin: 3px 0 6px 0; padding: 4px 8px;">
    <div class="title">High-Level Logic Summary: How do we generate lattices and guarantee they never fuse?</div>
    We don't draw every cage from scratch. We define a single prototype cage at the origin and clone it across space using <strong>SE(3) rigid transformation matrices</strong> (position + orientation). This generates 500+ cages in under 1 second. To guarantee cages never touch during 3D printing, we use a <strong>two-tier clearance engine</strong>: a fast spatial bounding-sphere filter (cKDTree) prunes 90% of non-neighbor pairs, and vectorized 3D geometry finds the exact closest approach between all remaining struts. Clearance scales linearly with cell size: <code>d<sub>min</sub> = &kappa; &middot; a</code>. If a simulation or manufacturing engineer specifies a desired strut diameter and clearance, our <strong>closed-form pitch solver</strong> calculates the exact required unit cell pitch in one formula.
  </div>

  <h3 style="margin-top: 5px; margin-bottom: 2px;">3.1 Memory-Efficient SE(3) Rigid Particle Abstraction</h3>
  <p style="margin-bottom: 4px;">
    Each particle is an independent <code>InterlinkedParticle</code> instance governed by an SE(3) matrix:
    <code>T = [ R | t ], R &in; SO(3), t &in; &Ropf;<sup>3</sup> &implies; V<sub>global</sub> = V &middot; R<sup>T</sup> + t<sup>T</sup></code>. Vertices and faces are evaluated lazily, allowing Graphite to assemble thousands of kinematic bodies with minimal memory and zero vertex-merging bugs.
  </p>

  <h3 style="margin-top: 5px; margin-bottom: 2px;">3.2 Two-Tier Clearance Verification Pipeline</h3>
  <ol style="margin-bottom: 4px;">
    <li style="margin-bottom: 1px;"><strong>Tier 1 (Broad Phase):</strong> Evaluates a spatial <code>scipy.spatial.cKDTree</code> across cage centroids. Only pairs satisfying <code>dist(c<sub>A</sub>, c<sub>B</sub>) &lt; 2.2 &middot; R<sub>outer</sub> + 2r</code> are retained (pruning &gt;90% in O(N log N) time).</li>
    <li style="margin-bottom: 1px;"><strong>Tier 2 (Narrow Phase):</strong> For retained pairs, computes the exact 3D segment Euclidean distance: <code>d(s<sub>A</sub>, s<sub>B</sub>) = min || p<sub>A</sub>(t<sub>A</sub>) &minus; p<sub>B</sub>(t<sub>B</sub>) ||</code>.</li>
    <li style="margin-bottom: 1px;"><strong>Surface Clearance Evaluation:</strong> Physical clearance is <code>&Delta; = d<sub>min</sub> &minus; D</code>. The DfAM guardrail halts execution if &Delta; &le; 0, preventing accidental solid collisions before exporting meshes.</li>
  </ol>

  <h3 style="margin-top: 5px; margin-bottom: 2px;">3.3 Crystallographic Clearance Constants &amp; Closed-Form Pitch Inversion</h3>
  <p style="margin-bottom: 3px;">
    Because cage geometry scales proportionally with unit cell pitch <code>a</code>, the minimum centerline distance satisfies a strict linear scaling law: <code>d<sub>min</sub> = &kappa; &middot; a</code>.
  </p>

  <table style="margin: 3px 0 5px 0;">
    <thead>
      <tr>
        <th style="padding: 3px 5px;">Lattice Topology</th>
        <th style="padding: 3px 5px;">Network Symmetry</th>
        <th style="padding: 3px 5px;">Characteristic Pitch Metric (a)</th>
        <th style="padding: 3px 5px;">Clearance Constant (&kappa;)</th>
        <th style="padding: 3px 5px;">Governing Clearance Law</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding: 3px 5px;"><strong>C-6-TT</strong></td>
        <td style="padding: 3px 5px;">Simple Cubic (<code>pcu</code>)</td>
        <td style="padding: 3px 5px;">Lattice repeat pitch a<sub>0</sub></td>
        <td style="padding: 3px 5px;"><span class="highlight">&kappa; &approx; 0.1314</span></td>
        <td style="padding: 3px 5px;">&Delta; = 0.1314 &middot; a<sub>0</sub> &minus; D</td>
      </tr>
      <tr>
        <td style="padding: 3px 5px;"><strong>D-4-TET (Conv)</strong></td>
        <td style="padding: 3px 5px;">Diamond Cubic (<code>dia</code>)</td>
        <td style="padding: 3px 5px;">Conventional cube size a<sub>conv</sub></td>
        <td style="padding: 3px 5px;"><span class="highlight">&kappa; &approx; 0.1018</span></td>
        <td style="padding: 3px 5px;">&Delta; = 0.1018 &middot; a<sub>conv</sub> &minus; D</td>
      </tr>
      <tr>
        <td style="padding: 3px 5px;"><strong>D-4-TET (Bond)</strong></td>
        <td style="padding: 3px 5px;">Diamond Cubic (<code>dia</code>)</td>
        <td style="padding: 3px 5px;">Bond length d<sub>bond</sub></td>
        <td style="padding: 3px 5px;"><span class="highlight">&kappa; &approx; 0.2354</span></td>
        <td style="padding: 3px 5px;">&Delta; = 0.2354 &middot; d<sub>bond</sub> &minus; D</td>
      </tr>
      <tr>
        <td style="padding: 3px 5px;"><strong>European 4-in-1</strong></td>
        <td style="padding: 3px 5px;">Planar Maille Weave</td>
        <td style="padding: 3px 5px;">Ring center pitch a<sub>ring</sub></td>
        <td style="padding: 3px 5px;"><span class="highlight">&kappa; &approx; 0.2200</span></td>
        <td style="padding: 3px 5px;">&Delta; = 0.2200 &middot; a<sub>ring</sub> &minus; D</td>
      </tr>
    </tbody>
  </table>

  <div class="sim-box" style="margin-top: 3px; margin-bottom: 0; padding: 4px 8px;">
    <div class="title">Closed-Form Pitch Solver for Simulation &amp; Manufacturing</div>
    Given target strut diameter <code>D</code> and required minimum manufacturing clearance <code>&Delta;<sub>target</sub></code>:
    <code>a<sub>required</sub> = (&Delta;<sub>target</sub> + D) / &kappa;</code>.
    <strong>Example:</strong> For C-6-TT with D = 0.50 mm and &Delta; = 200 &mu;m (0.20 mm):
    <code>a<sub>0</sub> = (0.20 + 0.50) / 0.1314 = 5.327 mm</code>. The engine automatically resizes the lattice to hit this tolerance.
  </div>
</div>

<!-- ================= PAGE 3 ================= -->
<div class="page-break"></div>
<div class="page-container">
  <h2>4. Functionally Graded Struts &amp; Continuous Topological Transitions</h2>

  <div class="logic-box">
    <div class="title">High-Level Logic Summary: Can we vary thickness or blend different cell types?</div>
    Yes! We can smoothly vary strut thickness or morph between entirely different cell geometries across the material without ever breaking the interlocking chainmail or causing cages to collide. For <strong>thickness grading</strong>, strut radius increases smoothly along an axis; our engine checks the thickest interface to guarantee clearance. For <strong>C6-to-D4 transitions</strong>, we introduce an edge truncation parameter &tau; &in; [0, 1/3]. Sweeping &tau; layer-by-layer shrinks the corner triangles of Truncated Tetrahedra until they collapse into sharp vertices of Regular Tetrahedra. The cages remain kinematically intertwined through the entire transition zone.
  </div>

  <h3>4.1 Functionally Graded PAMs (Directional Thickness Gradient)</h3>
  <p>
    Graphite allows continuous spatial variation of the strut radius <code>r(x)</code>:
    <span class="formula-card">r(X) = r<sub>min</sub> + (r<sub>max</sub> &minus; r<sub>min</sub>) &middot; [ (X &minus; X<sub>min</sub>) / (X<sub>max</sub> &minus; X<sub>min</sub>) ]</span>
    For adjacent particles A and B along the gradient, clearance is governed by:
    <code>&Delta;(X) = d<sub>min</sub> &minus; [ r(X) + r(X + a<sub>0</sub>) ] &ge; &Delta;<sub>target</sub> &gt; 0</code>.
    The engine verifies that the thickest end preserves printable clearance (&Delta; &ge; 514 &mu;m in Figure 4).
  </p>

  <div class="figure-container">
    <img src="{img_graded}" alt="Functionally Graded C6-TT PAM" style="max-height: 195px;">
    <div class="figure-caption"><strong>Figure 4:</strong> Functionally graded C-6-TT PAM (5 &times; 3 &times; 2, 30 cages) with strut radius graded from r = 0.22 mm (thin, teal) to r = 0.58 mm (thick, orange) with global min clearance &Delta; = 514 &mu;m.</div>
  </div>

  <h3>4.2 Topological &amp; Morphological Transitions: C-6-TT &harr; D-4-TET</h3>
  <p>
    Rather than welding disparate cell domains at a planar joint, Graphite morphs the underlying polyhedral cages via a continuous vertex truncation parameter <code>&tau; &in; [0, 1/3]</code>:
  </p>
  <div class="formula-card">P<sub>ij</sub> = (1 &minus; &tau;) V<sub>i</sub> + &tau; V<sub>j</sub>, &nbsp;&nbsp; P<sub>ji</sub> = &tau; V<sub>i</sub> + (1 &minus; &tau;) V<sub>j</sub></div>
  <ul>
    <li><strong>&tau; = 1/3 &approx; 0.333 (Archimedean Truncated Tetrahedron):</strong> 12 vertices, 18 struts, 4 hexagonal faces, 4 triangular corner windows. 6-fold coordination (C-6-TT).</li>
    <li><strong>0 &lt; &tau; &lt; 1/3 (Continuous Morphing Intermediate Cages):</strong> 12 vertices, 18 struts. Corner cutouts shrink continuously as &tau; decreases; hexagonal faces evolve toward triangular facets.</li>
    <li><strong>&tau; = 0.000 (Platonic Regular Tetrahedron):</strong> Truncated corner vertices converge to single points (P<sub>ij</sub> &rarr; V<sub>i</sub>). Exact mathematical reduction to 4 vertices and 6 struts. 4-fold coordination (D-4-TET).</li>
  </ul>

  <div class="figure-container">
    <img src="{img_morph}" alt="Continuous C6 to D4 Transition PAM" style="max-height: 200px;">
    <div class="figure-caption"><strong>Figure 5:</strong> Continuous 6-layer morphing lattice (6 &times; 2 &times; 2 = 24 cages) transitioning from pure C-6-TT (Teal, &tau; = 0.333) to pure D-4-TET (Crimson, &tau; = 0.000) with verified clearance &Delta; = +614 &mu;m.</div>
  </div>
</div>

<!-- ================= PAGE 4 ================= -->
<div class="page-break"></div>
<div class="page-container">
  <h2>5. Flagship Production Case Study: 75mm &times; 75mm &times; 20mm Transition Block</h2>

  <div class="logic-box">
    <div class="title">High-Level Logic Summary: Industrial-scale demonstration and exact clearance limits</div>
    To prove that this technology works at real-world manufacturing scale, we generated a full <strong>75 &times; 75 &times; 20 mm</strong> block containing <strong>540 discrete interlocking cages</strong> (678,960 watertight faces). The left 25 mm is pure C-6-TT, the center 25 mm continuously morphs from C6 to D4, and the right 25 mm is pure D-4-TET. We solved the exact centerline distance across all 2,955 interacting cage pairs: <strong>d<sub>min</sub> = 593.1 &mu;m</strong>. This proves mathematically that the thickest printable struts are <strong>443.1 &mu;m</strong> for 150 &mu;m clearance (LPBF) and <strong>393.1 &mu;m</strong> for 200 &mu;m clearance (SLS), with zero collisions. The entire 32 MB multi-body solid was generated in just <strong>6.1 seconds</strong>.
  </div>

  <div class="figure-container" style="margin-top: 3px;">
    <img src="{img_75}" alt="75mm x 75mm x 20mm PAM Transition Overview" style="max-height: 410px;">
    <div class="figure-caption"><strong>Figure 6:</strong> Comprehensive engineering overview of the 75 &times; 75 &times; 20 mm C6 &rarr; D4 transition PAM. Panel A: Full 3D isometric render (540 cages). Panel B: High-magnification cluster cutaways. Panel C: Clearance vs. cell pitch curves. Panel D: Engineering specification table.</div>
  </div>

  <h3>5.1 Analytical Clearance Limits &amp; Maximum Printable Strut Thickness</h3>
  <p>
    Across the 540-cage assembly (pitch a<sub>x</sub> = 5.0 mm, a<sub>y</sub> = 6.25 mm, a<sub>z</sub> = 6.67 mm, cage size s = 4.0 mm), the global minimum Euclidean distance between cage centerlines is <strong>d<sub>min</sub> = 0.5931 mm = 593.1 &mu;m</strong>.
  </p>
  <table>
    <thead>
      <tr>
        <th>Target Clearance (&Delta;<sub>target</sub>)</th>
        <th>Thickest Strut Diameter (D<sub>max</sub>)</th>
        <th>Thickest Strut Radius (r<sub>max</sub>)</th>
        <th>Verified Min Clearance</th>
        <th>Candidate Pairs Checked</th>
        <th>Recommended 3D Printing Process</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>150 &mu;m (0.150 mm)</strong></td>
        <td><span class="highlight">443.1 &mu;m (&approx; 0.44 mm)</span></td>
        <td>221.5 &mu;m (&approx; 0.22 mm)</td>
        <td>150.1 &mu;m</td>
        <td>2,955 pairs (0 collisions)</td>
        <td>LPBF / SLM (Ti-6Al-4V, 316L Stainless)</td>
      </tr>
      <tr>
        <td><strong>200 &mu;m (0.200 mm)</strong></td>
        <td><span class="highlight">393.1 &mu;m (&approx; 0.39 mm)</span></td>
        <td>196.5 &mu;m (&approx; 0.20 mm)</td>
        <td>200.0 &mu;m</td>
        <td>2,955 pairs (0 collisions)</td>
        <td>SLS (PA12 / Nylon) / DLP / PolyJet</td>
      </tr>
    </tbody>
  </table>
</div>

<!-- ================= PAGE 5 ================= -->
<div class="page-break"></div>
<div class="page-container">
  <h2>6. Additive Manufacturing Support Strategy: Seed Cell + Neighbor Ring</h2>

  <div class="logic-box">
    <div class="title">High-Level Logic Summary: How do we 3D print without fusing the cages with supports?</div>
    In metal powder-bed fusion (LPBF), downward-facing struts require print supports to conduct heat and prevent warping. But if you let commercial slicers generate supports automatically, they add thousands of support columns that bridge the tiny air gaps between cages, fusing the entire metamaterial into a solid block! Our solution: <strong>support a single central seed cell surrounded by its immediate coordination neighbor ring</strong>. We place tiny breakaway anchor pins only on downward nodes with thin conical necks (0.25&ndash;0.35 mm). Then our software automatically tessellates that support recipe across the entire lattice. Supports never bridge the gaps between cages, and loose powder evacuates freely through the open windows.
  </div>

  <div class="grid-2">
    <div class="figure-container">
      <img src="{img_supp}" alt="Interlinked Support Recipe Preview" style="max-height: 125px;">
      <div class="figure-caption"><strong>Figure 7:</strong> Central seed cell + coordination ring support recipe in Graphite Studio with breakaway anchor tessellation.</div>
    </div>
    <div class="figure-container">
      <img src="{img_studio}" alt="Graphite Trame Studio" style="max-height: 125px;">
      <div class="figure-caption"><strong>Figure 8:</strong> Interactive Trame Studio UI providing real-time 3D slicing, clearance verification, and parameter tuning.</div>
    </div>
  </div>

  <h2>7. Guidelines for FEA &amp; Multi-Body Dynamics (MBD) Simulation Engineers</h2>

  <div class="sim-box">
    <div class="title">Simulation Handoff: Setting up PAM Models in Abaqus, LS-DYNA, or Ansys</div>
    <ul>
      <li><strong>Multi-Body Part Topology:</strong> The exported STL / 3MF files contain <em>N distinct, closed, unbonded shells</em>. In your solver, do not merge vertices! Each shell must be imported as an independent body or part instance. For rigid-body kinematics (MBD), represent each cage as a 6-DOF rigid body with its mass, centroid, and inertia tensor.</li>
      <li><strong>Contact Formulation:</strong> Use general surface-to-surface or segment-to-segment penalty contact. Set the normal penalty stiffness <code>K<sub>n</sub> &approx; (E &middot; A) / L<sub>char</sub></code> high enough to prevent interpenetration (&delta;<sub>pen</sub> &lt; 0.05 &middot; r) while avoiding high-frequency chatter in explicit time integration.</li>
      <li><strong>Coulomb Friction is the Dominant Physics:</strong> Unlike bonded trusses that absorb energy through plastic strut yielding, PAMs absorb shock energy via sliding friction between cages. Setting accurate friction coefficients (as-built LPBF Ti-64: &mu; &approx; 0.30&ndash;0.40; SLS PA12: &mu; &approx; 0.15&ndash;0.25) is critical to capturing macroscopic damping and hysteresis loops.</li>
      <li><strong>Two-Phase Load-Displacement Response:</strong> Expect a flat "toe" region of kinematic free-play where cages rotate and settle, followed by a dramatic stiffening knee when contact locking occurs.</li>
      <li><strong>RVE Size for Homogenization:</strong> A minimum Representative Volume Element (RVE) of <code>2 &times; 2 &times; 2</code> conventional cells is recommended for periodic homogenization.</li>
    </ul>
  </div>

  <h2>8. Proven Verification Benchmarks &amp; Performance Summary</h2>

  <p>All models below have been synthesized, collision-checked across every interacting pair, and exported as watertight solids:</p>

  <table>
    <thead>
      <tr>
        <th>Model Benchmark</th>
        <th>Grid Dimensions</th>
        <th>Physical Envelope</th>
        <th>Cage Count</th>
        <th>Strut Diam (D)</th>
        <th>Verified Min Clearance</th>
        <th>Watertight Faces</th>
        <th>Gen Time</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>C-6-TT Clean Strip</strong></td>
        <td>15 &times; 5 &times; 4</td>
        <td>75 &times; 25 &times; 20 mm</td>
        <td>300</td>
        <td>0.50 mm (500 &mu;m)</td>
        <td><span class="highlight">156.9 &mu;m</span></td>
        <td>374,400</td>
        <td>2.8 s</td>
      </tr>
      <tr>
        <td><strong>D-4-TET Clean Strip</strong></td>
        <td>7 &times; 2 &times; 2 (conv)</td>
        <td>80.8 &times; 23.1 &times; 23.1 mm</td>
        <td>112</td>
        <td>0.75 mm (750 &mu;m)</td>
        <td><span class="highlight">427.0 &mu;m</span></td>
        <td>71,680</td>
        <td>1.1 s</td>
      </tr>
      <tr>
        <td><strong>C-6-TT Macroscopic Cube</strong></td>
        <td>3 &times; 3 &times; 3</td>
        <td>38.1 &times; 38.1 &times; 38.1 mm</td>
        <td>27</td>
        <td>1.25 mm (1250 &mu;m)</td>
        <td><span class="highlight">418.4 &mu;m</span></td>
        <td>58,320</td>
        <td>0.4 s</td>
      </tr>
      <tr>
        <td><strong>Graded C-6-TT PAM</strong></td>
        <td>5 &times; 3 &times; 2</td>
        <td>40 &times; 20 &times; 10 mm</td>
        <td>30</td>
        <td>0.44 &rarr; 1.16 mm</td>
        <td><span class="highlight">514.0 &mu;m</span></td>
        <td>37,440</td>
        <td>0.8 s</td>
      </tr>
      <tr>
        <td><strong>Morphing C6 &rarr; D4 Strip</strong></td>
        <td>6 &times; 2 &times; 2</td>
        <td>50 &times; 13 &times; 13 mm</td>
        <td>24</td>
        <td>0.70 mm (700 &mu;m)</td>
        <td><span class="highlight">614.0 &mu;m</span></td>
        <td>39,360</td>
        <td>0.6 s</td>
      </tr>
      <tr>
        <td><strong>Flagship 75 &times; 75 &times; 20 mm Block</strong></td>
        <td><strong>15 &times; 12 &times; 3</strong></td>
        <td><strong>75.0 &times; 75.0 &times; 20.0 mm</strong></td>
        <td><strong>540</strong></td>
        <td><strong>0.443 mm (443 &mu;m)</strong></td>
        <td><span class="highlight">150.1 &mu;m</span></td>
        <td><strong>678,960</strong></td>
        <td><strong>6.11 s</strong></td>
      </tr>
    </tbody>
  </table>

  <div class="meta-bar" style="margin-top: 10px; margin-bottom: 0;">
    <span><strong>Core Engine Entrypoint:</strong> <code>graphite/explicit/interlinked/</code></span>
    <span><strong>CAD Deliverables:</strong> <code>outputs/pam_transition_75x75x20_c6_to_d4.stl</code> (32.38 MB) &amp; <code>.3mf</code> (8.63 MB)</span>
  </div>
</div>

</body>
</html>
"""
    return html


def main() -> int:
    out_html = _OUTPUTS_DIR / "graphite_interlinked_pam_report.html"
    out_pdf = _OUTPUTS_DIR / "graphite_interlinked_pam_report.pdf"
    artifact_pdf = _ARTIFACT_DIR / "graphite_interlinked_pam_report.pdf"

    print("=================================================================")
    print("Generating Graphite Interlinked PAM Simulation Handoff Report PDF")
    print("=================================================================")

    # Step 1: Render HTML
    print("  [1/3] Generating HTML with embedded base64 figures...")
    html_content = build_html()
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"        Written: {out_html} ({len(html_content):,} chars)")

    # Step 2: Print to PDF via Headless Google Chrome
    print("  [2/3] Printing PDF via Headless Google Chrome...")
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        chrome_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

    cmd = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={out_pdf}",
        str(out_html),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error compiling PDF: {res.stderr}")
        return 1

    pdf_size_mb = os.path.getsize(out_pdf) / (1024 * 1024)
    print(f"        Generated PDF: {out_pdf} ({pdf_size_mb:.2f} MB)")

    # Step 3: Copy to Artifact Directory
    print("  [3/3] Copying PDF to Artifact Directory...")
    shutil.copy2(out_pdf, artifact_pdf)
    print(f"        Copied to: {artifact_pdf}")

    print("\n[Done] Handoff Report PDF successfully generated and deployed!")
    print("=================================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
