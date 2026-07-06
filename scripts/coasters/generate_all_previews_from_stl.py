import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt

COASTERS_DIR = r"C:\Users\ehunt\OneDrive\Documents\Python Scripts\Graphite\outputs\Coasters"
ARTIFACTS_DIR = r"C:\Users\ehunt\OneDrive\Documents\Python Scripts\Graphite\outputs\Coasters"

categories = {
    "TPMS": {
        "title": "TPMS 2D-preview (Z = 0 & Z = 1.25mm)",
        "files": [
            ("Gyroid", os.path.join(COASTERS_DIR, "TPMS", "Gyroid", "Circle", "Gyroid_Framed_Circle.stl")),
            ("Diamond", os.path.join(COASTERS_DIR, "TPMS", "Diamond", "Circle", "Diamond_Framed_Circle.stl")),
            ("Lidinoid", os.path.join(COASTERS_DIR, "TPMS", "Lidinoid", "Circle", "Lidinoid_Framed_Circle.stl")),
            ("Neovius", os.path.join(COASTERS_DIR, "TPMS", "Neovius", "Circle", "Neovius_Framed_Circle.stl")),
            ("Split-P", os.path.join(COASTERS_DIR, "TPMS", "Split-P", "Circle", "Split-P_Framed_Circle.stl")),
        ],
        "filename": "tpms_previews.png"
    },
    "Voroni Sparse": {
        "title": "Voroni Sparse 2D-preview (Z = 0)",
        "files": [
            ("large_v1", os.path.join(COASTERS_DIR, "Struts", "Voronoi", "Circle", "large_v1_Framed_Circle.stl")),
            ("large_v2", os.path.join(COASTERS_DIR, "Struts", "Voronoi", "Circle", "large_v2_Framed_Circle.stl")),
            ("large_v3", os.path.join(COASTERS_DIR, "Struts", "Voronoi", "Circle", "large_v3_Framed_Circle.stl")),
            ("large_v4", os.path.join(COASTERS_DIR, "Struts", "Voronoi", "Circle", "large_v4_Framed_Circle.stl")),
        ],
        "filename": "voronoi_sparse_previews.png"
    },
    "Voroni Dense": {
        "title": "Voroni Dense 2D-preview (Z = 0)",
        "files": [
            ("small_v1", os.path.join(COASTERS_DIR, "Struts", "Voronoi", "Circle", "small_v1_Framed_Circle.stl")),
            ("small_v2", os.path.join(COASTERS_DIR, "Struts", "Voronoi", "Circle", "small_v2_Framed_Circle.stl")),
            ("small_v3", os.path.join(COASTERS_DIR, "Struts", "Voronoi", "Circle", "small_v3_Framed_Circle.stl")),
        ],
        "filename": "voronoi_dense_previews.png"
    },
    "Explicit Tri": {
        "title": "Explicit Tri 2D-preview (Z = 0)",
        "files": [
            ("Tri_Icosahedral", os.path.join(COASTERS_DIR, "Struts", "Tri_Icosahedral", "Circle", "Tri_Icosahedral_Framed_Circle.stl")),
            ("Tri_Kelvin", os.path.join(COASTERS_DIR, "Struts", "Tri_Kelvin", "Circle", "Tri_Kelvin_Framed_Circle.stl")),
            ("Tri_Rhombic", os.path.join(COASTERS_DIR, "Struts", "Tri_Rhombic", "Circle", "Tri_Rhombic_Framed_Circle.stl")),
            ("Tri_Tesseract", os.path.join(COASTERS_DIR, "Struts", "Tri_Tesseract", "Circle", "Tri_Tesseract_Framed_Circle.stl")),
            ("Tri_Tetrahedral", os.path.join(COASTERS_DIR, "Struts", "Tri_Tetrahedral", "Circle", "Tri_Tetrahedral_Framed_Circle.stl")),
        ],
        "filename": "explicit_tri_previews.png"
    },
    "Explicit Square": {
        "title": "Explicit Square 2D-preview (Z = 0)",
        "files": [
            ("Sq_Grid", os.path.join(COASTERS_DIR, "Struts", "Sq_Grid", "Circle", "Sq_Grid_Framed_Circle.stl")),
            ("Sq_Icosahedral", os.path.join(COASTERS_DIR, "Struts", "Sq_Icosahedral", "Circle", "Sq_Icosahedral_Framed_Circle.stl")),
            ("Sq_Kelvin", os.path.join(COASTERS_DIR, "Struts", "Sq_Kelvin", "Circle", "Sq_Kelvin_Framed_Circle.stl")),
            ("Sq_Tesseract", os.path.join(COASTERS_DIR, "Struts", "Sq_Tesseract", "Circle", "Sq_Tesseract_Framed_Circle.stl")),
        ],
        "filename": "explicit_square_previews.png"
    },
    "A15": {
        "title": "A15 2D-preview (Z = 0)",
        "files": [
            ("A15_v1", os.path.join(COASTERS_DIR, "Struts", "A15", "Circle", "v1_Framed_Circle.stl")),
            ("A15_v2", os.path.join(COASTERS_DIR, "Struts", "A15", "Circle", "v2_Framed_Circle.stl")),
            ("A15_v3", os.path.join(COASTERS_DIR, "Struts", "A15", "Circle", "v3_Framed_Circle.stl")),
        ],
        "filename": "a15_previews.png"
    },
    "C15": {
        "title": "C15 2D-preview (Z = 0)",
        "files": [
            ("C15_v1", os.path.join(COASTERS_DIR, "Struts", "C15", "Circle", "v1_Framed_Circle.stl")),
            ("C15_v2", os.path.join(COASTERS_DIR, "Struts", "C15", "Circle", "v2_Framed_Circle.stl")),
            ("C15_v3", os.path.join(COASTERS_DIR, "Struts", "C15", "Circle", "v3_Framed_Circle.stl")),
        ],
        "filename": "c15_previews.png"
    }
}

def plot_section_outlines(ax, section, color='#1e3a8a'):
    for line in section.discrete:
        ax.plot(line[:, 0], line[:, 1], color=color, linewidth=1.2)

def generate_all_plots():
    for cat_name, cat_info in categories.items():
        print(f"Generating plot for {cat_name} from STL files...")
        files = cat_info["files"]
        N = len(files)
        
        if cat_name == "TPMS":
            # 2 rows for TPMS (Z = 0 and Z = 1.25)
            fig, axes = plt.subplots(2, N, figsize=(4 * N, 10))
            fig.subplots_adjust(hspace=0.4)
            fig.suptitle("TPMS 2D-preview (Row 1: Z=0, Row 2: Z=2.4mm)", fontsize=16, fontweight='bold', color='#0f172a', y=0.99)
            
            for idx, (label, path) in enumerate(files):
                # Row 1: Z = 0
                ax_top = axes[0, idx]
                ax_top.set_aspect('equal')
                ax_top.set_xlim(-55, 55)
                ax_top.set_ylim(-55, 55)
                ax_top.axis('off')
                ax_top.set_title(f"{label} (Z=0.0)", fontsize=12, fontweight='semibold', color='#334155', pad=10)
                
                # Row 2: Z = 1.25
                ax_bot = axes[1, idx]
                ax_bot.set_aspect('equal')
                ax_bot.set_xlim(-55, 55)
                ax_bot.set_ylim(-55, 55)
                ax_bot.axis('off')
                ax_bot.set_title(f"{label} (Z=2.4)", fontsize=12, fontweight='semibold', color='#334155', pad=4)
                
                if not os.path.exists(path):
                    ax_top.text(0, 0, "[File Not Found]", ha='center', va='center', color='red')
                    ax_bot.text(0, 0, "[File Not Found]", ha='center', va='center', color='red')
                    continue
                    
                try:
                    mesh = trimesh.load_mesh(path)
                    
                    # Top slice (Z = 0)
                    sec_top = mesh.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
                    if sec_top is not None:
                        plot_section_outlines(ax_top, sec_top)
                    else:
                        ax_top.text(0, 0, "[Empty Section]", ha='center', va='center', color='orange')
                        
                    # Bottom slice (Z = 1.25)
                    sec_bot = mesh.section(plane_origin=[0, 0, 2.4], plane_normal=[0, 0, 1])
                    if sec_bot is not None:
                        plot_section_outlines(ax_bot, sec_bot)
                    else:
                        ax_bot.text(0, 0, "[Empty Section]", ha='center', va='center', color='orange')
                        
                except Exception as e:
                    print(f"  Error processing TPMS {label}: {e}")
                    ax_top.text(0, 0, "[Error]", ha='center', va='center', color='red')
                    ax_bot.text(0, 0, "[Error]", ha='center', va='center', color='red')
            
        else:
            # 1 row for all other categories
            fig, axes = plt.subplots(1, N, figsize=(4 * N, 4.5))
            if N == 1:
                axes = [axes]
                
            fig.suptitle(cat_info["title"], fontsize=16, fontweight='bold', color='#0f172a', y=0.98)
            
            for idx, (label, path) in enumerate(files):
                ax = axes[idx]
                ax.set_aspect('equal')
                ax.set_xlim(-55, 55)
                ax.set_ylim(-55, 55)
                ax.axis('off')
                ax.set_title(label, fontsize=12, fontweight='semibold', color='#334155', pad=10)
                
                if not os.path.exists(path):
                    print(f"  Warning: file not found: {path}")
                    ax.text(0, 0, "[File Not Found]", ha='center', va='center', color='red')
                    continue
                    
                try:
                    mesh = trimesh.load_mesh(path)
                    
                    # Take cross-section at Z = 0
                    section = mesh.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
                    if section is not None:
                        plot_section_outlines(ax, section)
                    else:
                        print(f"  Warning: section at Z=0 is empty for {label}")
                        ax.text(0, 0, "[Empty Section]", ha='center', va='center', color='orange')
                            
                except Exception as e:
                    print(f"  Error processing {label}: {e}")
                    ax.text(0, 0, f"[Error]\n{str(e)[:20]}", ha='center', va='center', color='red', fontsize=8)
                    
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        out_path = os.path.join(ARTIFACTS_DIR, cat_info["filename"])
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    generate_all_plots()
