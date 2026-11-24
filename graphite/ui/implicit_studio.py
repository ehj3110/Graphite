"""Graphite Implicit Studio — Native PySide6 + VTK Desktop Workbench.

Midnight-slate 3-column workbench focused exclusively on implicit TPMS,
graded/piecewise architectures, woodpile structures, surface micro-features,
sub-second VTK surface previews, 2D 10 um slice inspection, calibration,
and production STL/3MF export.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PySide6.QtCore import QBuffer, QIODevice, QObject, QPoint, QPointF, QRectF, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
import trimesh
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from graphite.ui.implicit_help import SECTION_HELP_DOCS
from graphite.ui.implicit_state import (
    ALL_ARCHITECTURES,
    BLEND_METHODS,
    BLENDING_KERNELS,
    COORDINATE_AXES,
    DEFAULT_OUTPUT_ROOT,
    FOUR_WAY_METRICS,
    GRADING_MODES,
    GRADING_PRESETS,
    MULTI_KNOT_ENGINES,
    PILLAR_DISTRIBUTIONS,
    PILLAR_LOCATIONS,
    PREVIEW_MODES,
    PRIMITIVE_SHAPES,
    RAMP_AXES,
    RAMP_TARGETS,
    TEXTURE_DISPLACEMENT_MODES,
    TEXTURE_PROFILES,
    TEXTURE_TYPES,
    CalibrationOutcome,
    GenerationResult,
    ImplicitStudioState,
    build_or_load_cad_mesh,
    discover_cad_fixtures,
    estimate_voxel_budget,
    evaluate_grading_profile_1d,
    has_threemf_support,
    recommended_nyquist_resolution_mm,
    repair_cad_mesh_in_memory,
    run_implicit_generation,
    run_recipe_batch,
    run_tpms_calibration,
    solve_four_way_pair,
)
from graphite.ui.implicit_vtk_scene import (
    ImplicitVtkScene,
    build_micropillar_preview_polydata,
    build_surface_footprint_polydata,
    compute_2d_slice_image,
)


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0b111e;
    color: #e2e8f0;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 12px;
}
QDockWidget {
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    color: #38bdf8;
    font-weight: 600;
}
QDockWidget::title {
    background-color: #0f172a;
    padding: 6px 10px;
    border-bottom: 1px solid #1e293b;
}
QGroupBox {
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    margin-top: 12px;
    padding: 10px 8px 8px 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: #38bdf8;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
}
QPushButton {
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 5px 10px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #293548;
    border-color: #38bdf8;
}
QPushButton:pressed {
    background-color: #0f172a;
}
QPushButton:disabled {
    color: #64748b;
    background-color: #111827;
}
QPushButton#primaryBtn {
    background-color: #00c4a7;
    color: #041016;
    border: 1px solid #00f5d4;
    font-weight: 700;
    padding: 8px 14px;
}
QPushButton#primaryBtn:hover {
    background-color: #00f5d4;
}
QPushButton#accentBtn {
    background-color: #0c4a6e;
    color: #e0f2fe;
    border: 1px solid #38bdf8;
    font-weight: 600;
}
QPushButton#accentBtn:hover {
    background-color: #0369a1;
}
QPushButton#helpBtn {
    background-color: #1e293b;
    color: #38bdf8;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0px;
    font-weight: 700;
    font-size: 11px;
}
QPushButton#helpBtn:hover {
    background-color: #0c4a6e;
    color: #e0f2fe;
    border-color: #00f5d4;
}
QTextBrowser {
    background-color: #05070b;
    color: #e2e8f0;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 10px;
}
QComboBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit, QTableWidget {
    background-color: #0b111e;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 3px 6px;
    selection-background-color: #0284c7;
}
QDoubleSpinBox:disabled {
    background-color: #111827;
    color: #64748b;
    border-color: #1e293b;
}
QHeaderView::section {
    background-color: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
    padding: 3px;
    font-weight: 600;
}
QTabWidget::pane {
    border: 1px solid #1e293b;
    background-color: #05070b;
}
QTabBar::tab {
    background-color: #0f172a;
    color: #94a3b8;
    padding: 6px 14px;
    border: 1px solid #1e293b;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background-color: #1e293b;
    color: #00f5d4;
    font-weight: 600;
}
QToolBar {
    background-color: #0f172a;
    border-bottom: 1px solid #1e293b;
    spacing: 4px;
    padding: 3px 4px;
}
QProgressBar {
    background-color: #0b111e;
    border: 1px solid #334155;
    border-radius: 4px;
    text-align: center;
    color: #e2e8f0;
    height: 16px;
}
QProgressBar::chunk {
    background-color: #00f5d4;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #475569;
    background-color: #0b111e;
}
QCheckBox::indicator {
    border-radius: 3px;
}
QRadioButton::indicator {
    border-radius: 7px;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #00f5d4;
    border-color: #00f5d4;
}
"""


class _NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Spinbox that ignores mouse wheel events to preserve smooth dock scrolling."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class _HelpGroupBox(QGroupBox):
    """GroupBox with a compact floating '?' button aligned in the header bar."""

    def __init__(self, title: str, on_help_clicked, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.help_btn = QPushButton("?", self)
        self.help_btn.setObjectName("helpBtn")
        self.help_btn.setFixedSize(20, 20)
        self.help_btn.setCursor(Qt.PointingHandCursor)
        self.help_btn.setToolTip("Show section guide & parameter definitions")
        self.help_btn.clicked.connect(on_help_clicked)
        self.help_btn.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.help_btn.move(max(self.width() - self.help_btn.width() - 8, 4), 2)
        self.help_btn.raise_()


class _HelpPopupDialog(QDialog):
    """Midnight Slate scrollable documentation popup explaining section metrics & controls."""

    def __init__(self, title: str, html_body: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(560, 440)
        self.setStyleSheet(DARK_STYLESHEET)

        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(html_body)
        layout.addWidget(self.browser, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.setObjectName("accentBtn")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)


class _GradingProfileChartWidget(QWidget):
    """Pure QPainter Midnight-Slate 1D chart plotting L(s), w(s), and SF(s)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(260, 240)
        self._profile: dict[str, Any] | None = None
        self._slice_coord_mm: float | None = None
        self._slice_axis: str = "Z"

    def set_profile(
        self,
        profile: dict[str, Any],
        *,
        slice_coord_mm: float | None = None,
        slice_axis: str = "Z",
    ) -> None:
        self._profile = profile
        self._slice_coord_mm = slice_coord_mm
        self._slice_axis = slice_axis
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())
        painter.fillRect(self.rect(), QColor("#05070b"))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRect(QRectF(0.5, 0.5, max(w - 1.0, 1.0), max(h - 1.0, 1.0)))

        prof = self._profile
        if not prof or len(prof.get("pos_mm", [])) < 2:
            painter.setPen(QColor("#64748b"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No grading profile data")
            return

        pos = np.asarray(prof["pos_mm"], dtype=float)
        L_arr = np.asarray(prof["L_mm"], dtype=float)
        w_arr = np.asarray(prof["w_mm"], dtype=float)
        sf_arr = np.asarray(prof["sf"], dtype=float)
        axis_name = str(prof.get("axis", "Z"))
        markers = [float(m) for m in prof.get("markers_mm", [])]

        ml, mr, mt, mb = 52.0, 50.0, 34.0, 36.0
        pw = max(w - ml - mr, 40.0)
        ph = max(h - mt - mb, 40.0)
        plot_rect = QRectF(ml, mt, pw, ph)

        x_min, x_max = float(pos[0]), float(pos[-1])
        if abs(x_max - x_min) < 1e-6:
            x_max = x_min + 1.0

        mm_max = max(float(np.max(L_arr)), float(np.max(w_arr)), 1.0) * 1.15
        mm_min = 0.0
        sf_min, sf_max = 0.0, 1.0

        def map_x(val: float) -> float:
            return ml + ((val - x_min) / (x_max - x_min)) * pw

        def map_mm(val: float) -> float:
            return mt + ph - ((val - mm_min) / (mm_max - mm_min)) * ph

        def map_sf(val: float) -> float:
            t = max(0.0, min(1.0, (val - sf_min) / (sf_max - sf_min)))
            return mt + ph - t * ph

        # Subtle background grid
        painter.setPen(QPen(QColor("#162032"), 1.0, Qt.DotLine))
        for i in range(5):
            gy = mt + (i / 4.0) * ph
            gx = ml + (i / 4.0) * pw
            painter.drawLine(QPointF(ml, gy), QPointF(ml + pw, gy))
            painter.drawLine(QPointF(gx, mt), QPointF(gx, mt + ph))

        # Plot frame
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRect(plot_rect)

        font = QFont("Segoe UI", 8)
        painter.setFont(font)

        # Marker lines (knots / piecewise breaks)
        painter.setPen(QPen(QColor("#64748b"), 1.2, Qt.DashDotLine))
        for m in markers:
            if x_min <= m <= x_max:
                mx = map_x(m)
                painter.drawLine(QPointF(mx, mt), QPointF(mx, mt + ph))

        # Draw curves
        def draw_series(vals: np.ndarray, y_fn, color_hex: str, width: float) -> None:
            pen = QPen(QColor(color_hex), width)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            pts = [QPointF(map_x(float(pos[i])), y_fn(float(vals[i]))) for i in range(len(pos))]
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

        draw_series(L_arr, map_mm, "#00f5d4", 2.2)
        draw_series(w_arr, map_mm, "#fbbf24", 1.9)
        draw_series(sf_arr, map_sf, "#38bdf8", 2.2)

        # Active slice plane cursor when slice axis matches profile axis
        if self._slice_coord_mm is not None and self._slice_axis.upper() == axis_name.upper():
            if x_min <= self._slice_coord_mm <= x_max:
                sx = map_x(self._slice_coord_mm)
                painter.setPen(QPen(QColor("#f472b6"), 1.8, Qt.DashLine))
                painter.drawLine(QPointF(sx, mt), QPointF(sx, mt + ph))
                painter.setPen(QColor("#f472b6"))
                badge = f"Slice {self._slice_coord_mm:.2f} mm"
                tx = min(max(sx + 4.0, ml + 4.0), ml + pw - 96.0)
                painter.drawText(QPointF(tx, mt + 13.0), badge)

        # Left Y axis labels (mm)
        painter.setPen(QColor("#00f5d4"))
        for i in range(5):
            val = mm_max * (1.0 - i / 4.0)
            gy = mt + (i / 4.0) * ph
            painter.drawText(QRectF(2.0, gy - 8.0, ml - 6.0, 16.0), Qt.AlignRight | Qt.AlignVCenter, f"{val:.2f}")

        # Right Y axis labels (SF)
        painter.setPen(QColor("#38bdf8"))
        for i in range(5):
            val = 1.0 - i / 4.0
            gy = mt + (i / 4.0) * ph
            painter.drawText(QRectF(ml + pw + 5.0, gy - 8.0, mr - 7.0, 16.0), Qt.AlignLeft | Qt.AlignVCenter, f"{val:.2f}")

        # Bottom X axis labels
        painter.setPen(QColor("#94a3b8"))
        for i in range(5):
            val = x_min + (i / 4.0) * (x_max - x_min)
            gx = ml + (i / 4.0) * pw
            painter.drawText(QRectF(gx - 24.0, mt + ph + 3.0, 48.0, 15.0), Qt.AlignCenter, f"{val:.1f}")
        painter.drawText(
            QRectF(ml, mt + ph + 18.0, pw, 16.0),
            Qt.AlignCenter,
            f"Grading Axis: {axis_name} (mm)",
        )

        # Top legend strip (responsive spacing so labels never clip)
        legend = [
            ("#00f5d4", "L (mm)"),
            ("#fbbf24", "Wall w"),
            ("#38bdf8", "SF (0-1)"),
        ]
        lx = max(ml - 8.0, 8.0)
        step_x = max(min(84.0, (w - lx - 10.0) / 3.0), 62.0)
        for col, text in legend:
            painter.setPen(QPen(QColor(col), 2.5))
            painter.drawLine(QPointF(lx, 14.0), QPointF(lx + 12.0, 14.0))
            painter.setPen(QColor("#cbd5e1"))
            painter.drawText(QPointF(lx + 16.0, 18.0), text)
            lx += step_x



class _CalibrationWorker(QObject):
    """Background worker for exact LUT/Newton TPMS calibration."""

    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, state: ImplicitStudioState) -> None:
        super().__init__()
        self._state = state

    @Slot()
    def run(self) -> None:
        try:
            outcome = run_tpms_calibration(
                self._state, progress_cb=lambda msg: self.progress.emit(msg)
            )
            self.finished.emit(outcome)
        except Exception as exc:
            self.failed.emit(str(exc))


class _GenerationWorker(QObject):
    """Background worker for full 3D implicit lattice generation + export."""

    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, state: ImplicitStudioState, cad_mesh: trimesh.Trimesh | None) -> None:
        super().__init__()
        self._state = state
        self._cad_mesh = cad_mesh

    @Slot()
    def run(self) -> None:
        try:
            result = run_implicit_generation(
                self._state,
                cad_mesh=self._cad_mesh,
                progress_cb=lambda pct, msg: self.progress.emit(pct, msg),
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class _BatchWorker(QObject):
    """Background worker for sequential batch execution of saved JSON recipes."""

    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, recipe_paths: list[str | Path]) -> None:
        super().__init__()
        self._recipe_paths = list(recipe_paths)

    @Slot()
    def run(self) -> None:
        try:
            def _on_prog(idx: int, total: int, rpath: Path, msg: str) -> None:
                pct = int(round(100.0 * idx / max(total, 1)))
                self.progress.emit(pct, f"[{min(idx + 1, max(total, 1))}/{max(total, 1)} {rpath.name}] {msg}")

            out = run_recipe_batch(self._recipe_paths, progress_cb=_on_prog)
            self.finished.emit(out)
        except Exception as exc:
            self.failed.emit(str(exc))


class ImplicitStudioWindow(QMainWindow):
    """Main Midnight Slate 3-Column Workbench for Graphite Implicit Lattices."""

    def __init__(
        self,
        initial_state: ImplicitStudioState | None = None,
        *,
        enable_vtk: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Graphite Implicit Studio")
        self.resize(1540, 940)
        self.setStyleSheet(DARK_STYLESHEET)

        self.state = initial_state or ImplicitStudioState()
        self._cad_mesh: trimesh.Trimesh | None = None
        self._cad_health: dict[str, Any] = {}
        self._generated_mesh: trimesh.Trimesh | None = None
        self._last_result: GenerationResult | None = None
        self._updating_ui = False
        self._enable_vtk = enable_vtk

        self._cal_thread: QThread | None = None
        self._gen_thread: QThread | None = None
        self._batch_thread: QThread | None = None
        self._vtk_initialized = False
        self._cut_dragging = False
        self._help_boxes: dict[str, _HelpGroupBox] = {}

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(self.refresh_visual_previews)

        self._cutaway_cap_timer = QTimer(self)
        self._cutaway_cap_timer.setSingleShot(True)
        self._cutaway_cap_timer.setInterval(100)
        self._cutaway_cap_timer.timeout.connect(self._finalize_cutaway_cap)

        self._slice_timer = QTimer(self)
        self._slice_timer.setSingleShot(True)
        self._slice_timer.setInterval(40)
        self._slice_timer.timeout.connect(self._refresh_slice_tab)

        self._build_toolbar()
        self._build_central_viewport()
        self._build_left_dock()
        self._build_right_dock()
        self._bind_shortcuts()

        self._sync_widgets_from_state()
        self._rebuild_cad_and_refresh()

    # ------------------------------------------------------------------
    # Toolbar & Central Dual-Tab Viewport
    # ------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        tb = QToolBar("Viewport Controls", self)
        tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)

        for label, preset in (
            ("Iso (1)", "iso"),
            ("Top (2)", "top"),
            ("Front (3)", "front"),
            ("Side (4)", "side"),
            ("Reset (R)", "reset"),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _checked=False, p=preset: self._set_camera(p))
            tb.addWidget(btn)

        tb.addSeparator()
        tb.addWidget(QLabel(" Preview: "))
        self._combo_preview_mode = QComboBox()
        self._combo_preview_mode.addItems(list(PREVIEW_MODES))
        self._combo_preview_mode.setToolTip(
            "2D Surface: opaque gray CAD with sub-50 um 2D surface overlay & cutaway slice cap (~0.05 s)\n"
            "3D Flying Edges (Coarse): lightweight 3D preview isosurface (~350k voxel cap); full resolution deferred to export"
        )
        self._combo_preview_mode.currentTextChanged.connect(self._on_preview_mode_changed)
        tb.addWidget(self._combo_preview_mode)

        tb.addWidget(QLabel(" CAD: "))
        self._opacity_combo = QComboBox()
        self._opacity_combo.addItems(["100% Opaque Gray", "25% Ghost", "15% Faint", "0% Hidden"])
        self._opacity_combo.currentIndexChanged.connect(self._on_opacity_changed)
        tb.addWidget(self._opacity_combo)

        tb.addSeparator()
        self._chk_footprint = QCheckBox("Surface Footprint")
        self._chk_footprint.setChecked(True)
        self._chk_footprint.toggled.connect(self._on_footprint_toggled)
        tb.addWidget(self._chk_footprint)

        self._chk_show_gen_mesh_tb = QCheckBox("Show 3D Mesh")
        self._chk_show_gen_mesh_tb.setChecked(self.state.show_generated_3d_mesh)
        self._chk_show_gen_mesh_tb.toggled.connect(self._on_show_generated_toggled)
        tb.addWidget(self._chk_show_gen_mesh_tb)

        tb.addSeparator()
        self._chk_cutaway = QCheckBox("Live Cutaway")
        self._chk_cutaway.toggled.connect(self._on_cutaway_changed)
        tb.addWidget(self._chk_cutaway)

        self._combo_cut_axis = QComboBox()
        self._combo_cut_axis.addItems(["Z", "X", "Y"])
        self._combo_cut_axis.currentIndexChanged.connect(self._on_cutaway_changed)
        tb.addWidget(self._combo_cut_axis)

        self._slider_cut = QSlider(Qt.Horizontal)
        self._slider_cut.setRange(2, 98)
        self._slider_cut.setValue(50)
        self._slider_cut.setFixedWidth(96)
        self._slider_cut.sliderPressed.connect(self._on_cutaway_slider_pressed)
        self._slider_cut.sliderReleased.connect(self._on_cutaway_slider_released)
        self._slider_cut.valueChanged.connect(self._on_cutaway_value_changed)
        tb.addWidget(self._slider_cut)

        self._lbl_cut_pct = QLabel("50%")
        self._lbl_cut_pct.setFixedWidth(36)
        tb.addWidget(self._lbl_cut_pct)

        tb.addSeparator()
        self._btn_snapshot = QPushButton("Snapshot")
        self._btn_snapshot.setToolTip("Save window screenshot to outputs/implicit_studio/snapshots/ (Ctrl+P)")
        self._btn_snapshot.clicked.connect(self._on_snapshot_clicked)
        tb.addWidget(self._btn_snapshot)

        self._btn_how_to_use = QPushButton("How to Use (?)")
        self._btn_how_to_use.setObjectName("accentBtn")
        self._btn_how_to_use.setToolTip("Open Graphite Implicit Studio quick-start workflow & shortcuts")
        self._btn_how_to_use.clicked.connect(lambda: self._show_help_popup("quick_start"))
        tb.addWidget(self._btn_how_to_use)

    def _build_central_viewport(self) -> None:
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        # Tab 1: 3D VTK Viewport
        vtk_container = QWidget()
        vtk_layout = QVBoxLayout(vtk_container)
        vtk_layout.setContentsMargins(0, 0, 0, 0)
        self._vtk_widget: QVTKRenderWindowInteractor | None = None
        self._renderer: vtk.vtkRenderer | None = None
        self._scene: ImplicitVtkScene | None = None

        if self._enable_vtk:
            self._vtk_widget = QVTKRenderWindowInteractor(vtk_container)
            self._renderer = vtk.vtkRenderer()
            self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)
            self._scene = ImplicitVtkScene(self._renderer)
            vtk_layout.addWidget(self._vtk_widget)
        else:
            placeholder = QLabel("VTK Viewport disabled (headless test mode)")
            placeholder.setAlignment(Qt.AlignCenter)
            vtk_layout.addWidget(placeholder)

        self._tabs.addTab(vtk_container, "3D Studio Viewport")

        # Tab 2: 2D Cross-Section Slice Inspector
        slice_tab = QWidget()
        slice_layout = QVBoxLayout(slice_tab)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Slice Plane:"))
        self._combo_slice_axis = QComboBox()
        self._combo_slice_axis.addItems(["Z", "X", "Y"])
        self._combo_slice_axis.currentIndexChanged.connect(self._refresh_slice_tab)
        ctrl_row.addWidget(self._combo_slice_axis)

        ctrl_row.addWidget(QLabel("Position:"))
        self._slider_slice_pos = QSlider(Qt.Horizontal)
        self._slider_slice_pos.setRange(2, 98)
        self._slider_slice_pos.setValue(50)
        self._slider_slice_pos.valueChanged.connect(self._on_slice_slider_changed)
        ctrl_row.addWidget(self._slider_slice_pos, 1)

        self._lbl_slice_pos = QLabel("50%")
        ctrl_row.addWidget(self._lbl_slice_pos)

        ctrl_row.addWidget(QLabel("Pixel Pitch:"))
        self._combo_slice_px = QComboBox()
        self._combo_slice_px.addItems(["0.050 mm", "0.025 mm", "0.010 mm (10 um)", "0.100 mm"])
        self._combo_slice_px.currentIndexChanged.connect(self._refresh_slice_tab)
        ctrl_row.addWidget(self._combo_slice_px)

        btn_slice = QPushButton("Refresh Slice")
        btn_slice.clicked.connect(self._refresh_slice_tab)
        ctrl_row.addWidget(btn_slice)
        slice_layout.addLayout(ctrl_row)

        slice_split = QSplitter(Qt.Horizontal)

        left_slice_pane = QWidget()
        left_slice_v = QVBoxLayout(left_slice_pane)
        left_slice_v.setContentsMargins(0, 0, 4, 0)
        self._lbl_slice_image = QLabel("Switch to this tab or press Refresh Slice to render.")
        self._lbl_slice_image.setAlignment(Qt.AlignCenter)
        self._lbl_slice_image.setStyleSheet("background-color: #05070b; border: 1px solid #1e293b;")
        self._lbl_slice_image.setMinimumHeight(420)
        left_slice_v.addWidget(self._lbl_slice_image, 1)

        self._lbl_slice_metrics = QLabel("")
        self._lbl_slice_metrics.setWordWrap(True)
        self._lbl_slice_metrics.setStyleSheet("color: #38bdf8; font-family: Consolas, monospace; padding: 4px;")
        left_slice_v.addWidget(self._lbl_slice_metrics)

        right_chart_pane = QWidget()
        right_chart_v = QVBoxLayout(right_chart_pane)
        right_chart_v.setContentsMargins(4, 0, 0, 0)
        chart_hdr = QLabel("1D Spatial Grading & Wall/Cell Profile")
        chart_hdr.setStyleSheet("color: #00f5d4; font-weight: 600; padding: 2px 0;")
        right_chart_v.addWidget(chart_hdr)

        self._profile_chart = _GradingProfileChartWidget()
        right_chart_v.addWidget(self._profile_chart, 1)

        self._lbl_profile_summary = QLabel("Profile summary pending...")
        self._lbl_profile_summary.setWordWrap(True)
        self._lbl_profile_summary.setStyleSheet("color: #94a3b8; font-family: Consolas, monospace; padding: 4px;")
        right_chart_v.addWidget(self._lbl_profile_summary)

        slice_split.addWidget(left_slice_pane)
        slice_split.addWidget(right_chart_pane)
        slice_split.setSizes([640, 440])
        slice_layout.addWidget(slice_split, 1)

        self._tabs.addTab(slice_tab, "2D Cross-Section Slice")
        self._tabs.currentChanged.connect(self._on_tab_changed)

    def _make_help_group(self, section_key: str, title: str) -> _HelpGroupBox:
        grp = _HelpGroupBox(title, lambda _checked=False, k=section_key: self._show_help_popup(k))
        self._help_boxes[section_key] = grp
        return grp

    def _show_help_popup(self, section_key: str, *, exec_dialog: bool = True) -> _HelpPopupDialog:
        title, html_body = SECTION_HELP_DOCS[section_key]
        dlg = _HelpPopupDialog(title, html_body, self)
        if exec_dialog:
            dlg.exec()
        return dlg

    # ------------------------------------------------------------------
    # Left Setup Dock
    # ------------------------------------------------------------------
    def _build_left_dock(self) -> None:
        dock = QDockWidget("Lattice Setup", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock.setMinimumWidth(455)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(6)

        # 1. CAD Geometry & Health
        grp_cad = self._make_help_group("cad_geometry", "1. CAD Geometry && Health")
        cad_v = QVBoxLayout(grp_cad)

        src_row = QHBoxLayout()
        self._rb_prim = QRadioButton("Primitive")
        self._rb_file = QRadioButton("STL Fixture / File")
        self._rb_prim.toggled.connect(self._on_cad_source_changed)
        src_row.addWidget(self._rb_prim)
        src_row.addWidget(self._rb_file)
        cad_v.addLayout(src_row)

        self._cad_stack = QStackedWidget()
        # Primitive page
        prim_w = QWidget()
        prim_form = QFormLayout(prim_w)
        prim_form.setContentsMargins(0, 4, 0, 0)
        self._combo_prim_shape = QComboBox()
        self._combo_prim_shape.addItems(list(PRIMITIVE_SHAPES))
        self._combo_prim_shape.currentIndexChanged.connect(self._on_cad_param_changed)
        prim_form.addRow("Shape:", self._combo_prim_shape)

        dims_row = QHBoxLayout()
        self._spin_prim_x = self._make_spin(1.0, 500.0, 0.5, 1, " mm")
        self._spin_prim_y = self._make_spin(1.0, 500.0, 0.5, 1, " mm")
        self._spin_prim_z = self._make_spin(1.0, 500.0, 0.5, 1, " mm")
        for s in (self._spin_prim_x, self._spin_prim_y, self._spin_prim_z):
            s.valueChanged.connect(self._on_cad_param_changed)
            dims_row.addWidget(s)
        prim_form.addRow("X / Y / Z:", dims_row)
        self._cad_stack.addWidget(prim_w)

        # File page
        file_w = QWidget()
        file_v = QVBoxLayout(file_w)
        file_v.setContentsMargins(0, 4, 0, 0)
        self._combo_fixtures = QComboBox()
        for p in discover_cad_fixtures():
            self._combo_fixtures.addItem(p.name, str(p))
        self._combo_fixtures.currentIndexChanged.connect(self._on_fixture_selected)
        file_v.addWidget(self._combo_fixtures)

        browse_row = QHBoxLayout()
        self._edit_cad_path = QLineEdit()
        self._edit_cad_path.setPlaceholderText("Select an STL file...")
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_cad_stl)
        browse_row.addWidget(self._edit_cad_path, 1)
        browse_row.addWidget(btn_browse)
        file_v.addLayout(browse_row)
        self._cad_stack.addWidget(file_w)
        cad_v.addWidget(self._cad_stack)

        self._chk_center_origin = QCheckBox("Center CAD at Origin (0, 0, 0)")
        self._chk_center_origin.toggled.connect(self._on_cad_param_changed)
        cad_v.addWidget(self._chk_center_origin)

        health_row = QHBoxLayout()
        self._lbl_cad_health = QLabel("CAD: Pending...")
        self._lbl_cad_health.setWordWrap(True)
        self._btn_repair_cad = QPushButton("Repair")
        self._btn_repair_cad.setFixedWidth(70)
        self._btn_repair_cad.clicked.connect(self._on_repair_cad_clicked)
        health_row.addWidget(self._lbl_cad_health, 1)
        health_row.addWidget(self._btn_repair_cad)
        cad_v.addLayout(health_row)
        layout.addWidget(grp_cad)

        # 2. Architecture & Shelling
        grp_arch = self._make_help_group("architecture_shelling", "2. Architecture && Shelling")
        arch_form = QFormLayout(grp_arch)
        self._combo_arch = QComboBox()
        self._combo_arch.addItems(list(ALL_ARCHITECTURES))
        self._combo_arch.currentIndexChanged.connect(self._on_arch_changed)
        arch_form.addRow("Architecture:", self._combo_arch)

        self._chk_true_woodpile = QCheckBox("True Shifted Woodpile")
        self._chk_true_woodpile.toggled.connect(self._on_generic_param_changed)
        arch_form.addRow("", self._chk_true_woodpile)

        self._chk_generate_skin = QCheckBox("Generate Skin")
        self._chk_generate_skin.toggled.connect(self._on_generate_skin_toggled)
        arch_form.addRow("", self._chk_generate_skin)

        self._skin_options_widget = QWidget()
        skin_form = QFormLayout(self._skin_options_widget)
        skin_form.setContentsMargins(0, 0, 0, 0)

        shell_row = QHBoxLayout()
        self._spin_shell_t = self._make_spin(0.1, 20.0, 0.1, 2, " mm")
        self._spin_blend_r = self._make_spin(0.0, 10.0, 0.1, 2, " mm")
        self._spin_shell_t.valueChanged.connect(self._on_generic_param_changed)
        self._spin_blend_r.valueChanged.connect(self._on_generic_param_changed)
        shell_row.addWidget(QLabel("Skin:"))
        shell_row.addWidget(self._spin_shell_t)
        shell_row.addWidget(QLabel("Fillet:"))
        shell_row.addWidget(self._spin_blend_r)
        skin_form.addRow("Shell / Blend:", shell_row)

        self._combo_blend_method = QComboBox()
        self._combo_blend_method.addItems(list(BLEND_METHODS))
        self._combo_blend_method.currentIndexChanged.connect(self._on_generic_param_changed)
        skin_form.addRow("Blend Method:", self._combo_blend_method)

        self._skin_options_widget.setVisible(False)
        arch_form.addRow(self._skin_options_widget)
        layout.addWidget(grp_arch)

        # 3. 4-Way Sizing & Resolution
        grp_size = self._make_help_group("four_way_sizing", "3. 4-Way Sizing && Calibration")
        size_form = QFormLayout(grp_size)

        drv_row = QHBoxLayout()
        self._combo_size_drv = QComboBox()
        self._combo_size_drv.addItem("Select Driver 1...", "")
        self._combo_density_drv = QComboBox()
        self._combo_density_drv.addItem("Select Driver 2...", "")
        for key, label in FOUR_WAY_METRICS:
            self._combo_size_drv.addItem(label, key)
            self._combo_density_drv.addItem(label, key)
        self._combo_size_drv.currentIndexChanged.connect(lambda: self._on_drivers_changed(1))
        self._combo_density_drv.currentIndexChanged.connect(lambda: self._on_drivers_changed(2))
        drv_row.addWidget(self._combo_size_drv)
        drv_row.addWidget(self._combo_density_drv)
        size_form.addRow("Drivers:", drv_row)

        self._spin_L = self._make_spin(0.2, 50.0, 0.1, 3, " mm")
        self._spin_pore = self._make_spin(0.1, 40.0, 0.1, 3, " mm")
        self._spin_sf = self._make_spin(0.05, 0.90, 0.02, 3, "")
        self._spin_wall = self._make_spin(0.03, 10.0, 0.05, 3, " mm")
        self._spin_L.valueChanged.connect(lambda: self._on_four_way_changed("unit_cell"))
        self._spin_pore.valueChanged.connect(lambda: self._on_four_way_changed("pore_size"))
        self._spin_sf.valueChanged.connect(lambda: self._on_four_way_changed("solid_fraction"))
        self._spin_wall.valueChanged.connect(lambda: self._on_four_way_changed("wall_thickness"))

        size_form.addRow("Unit Cell L:", self._spin_L)
        size_form.addRow("Pore Size (MIS):", self._spin_pore)
        size_form.addRow("Solid Fraction:", self._spin_sf)
        size_form.addRow("Wall Thickness:", self._spin_wall)

        res_row = QHBoxLayout()
        self._chk_auto_res = QCheckBox("Auto Nyquist")
        self._chk_auto_res.toggled.connect(self._on_auto_res_toggled)
        self._spin_res = self._make_spin(0.01, 2.0, 0.01, 3, " mm")
        self._spin_res.valueChanged.connect(self._on_generic_param_changed)
        res_row.addWidget(self._chk_auto_res)
        res_row.addWidget(self._spin_res, 1)
        size_form.addRow("Voxel Pitch:", res_row)

        self._btn_calibrate = QPushButton("Calibrate Exact (LUT / Newton)")
        self._btn_calibrate.setObjectName("accentBtn")
        self._btn_calibrate.clicked.connect(self._on_calibrate_clicked)
        size_form.addRow(self._btn_calibrate)
        layout.addWidget(grp_size)

        # 4. Unified Functional Grading Mode
        grp_grade = self._make_help_group("functional_grading", "4. Unified Functional Grading")
        grade_v = QVBoxLayout(grp_grade)

        preset_row = QHBoxLayout()
        self._combo_preset = QComboBox()
        self._combo_preset.addItem("Apply Grading Preset...")
        self._combo_preset.addItems(list(GRADING_PRESETS))
        self._combo_preset.currentIndexChanged.connect(self._on_preset_selected)
        preset_row.addWidget(self._combo_preset)
        grade_v.addLayout(preset_row)

        self._combo_grading_mode = QComboBox()
        self._combo_grading_mode.addItems(list(GRADING_MODES))
        self._combo_grading_mode.currentIndexChanged.connect(self._on_grading_mode_changed)
        grade_v.addWidget(self._combo_grading_mode)

        self._grade_stack = QStackedWidget()

        # Page 0: Uniform + Rotation
        p_uni = QWidget()
        p_uni_form = QFormLayout(p_uni)
        p_uni_form.setContentsMargins(0, 4, 0, 0)
        rot_row = QHBoxLayout()
        self._spin_rx = self._make_spin(-180.0, 180.0, 5.0, 1, "°")
        self._spin_ry = self._make_spin(-180.0, 180.0, 5.0, 1, "°")
        self._spin_rz = self._make_spin(-180.0, 180.0, 5.0, 1, "°")
        for s in (self._spin_rx, self._spin_ry, self._spin_rz):
            s.valueChanged.connect(self._on_generic_param_changed)
            rot_row.addWidget(s)
        p_uni_form.addRow("Rotation X/Y/Z:", rot_row)
        self._grade_stack.addWidget(p_uni)

        # Page 1: Linear / Radial Ramp
        p_ramp = QWidget()
        p_ramp_form = QFormLayout(p_ramp)
        p_ramp_form.setContentsMargins(0, 4, 0, 0)
        self._combo_ramp_target = QComboBox()
        self._combo_ramp_target.addItems(list(RAMP_TARGETS))
        self._combo_ramp_axis = QComboBox()
        self._combo_ramp_axis.addItems(list(RAMP_AXES))
        self._spin_ramp_sf0 = self._make_spin(0.05, 0.90, 0.02, 2, "")
        self._spin_ramp_sf1 = self._make_spin(0.05, 0.90, 0.02, 2, "")
        self._spin_ramp_L0 = self._make_spin(0.3, 30.0, 0.2, 2, " mm")
        self._spin_ramp_L1 = self._make_spin(0.3, 30.0, 0.2, 2, " mm")
        self._spin_ramp_trans = self._make_spin(0.1, 100.0, 0.5, 2, " mm")
        for w in (
            self._combo_ramp_target,
            self._combo_ramp_axis,
            self._spin_ramp_sf0,
            self._spin_ramp_sf1,
            self._spin_ramp_L0,
            self._spin_ramp_L1,
            self._spin_ramp_trans,
        ):
            sig = w.currentIndexChanged if isinstance(w, QComboBox) else w.valueChanged
            sig.connect(self._on_generic_param_changed)
        sf_r = QHBoxLayout()
        sf_r.addWidget(self._spin_ramp_sf0)
        sf_r.addWidget(self._spin_ramp_sf1)
        l_r = QHBoxLayout()
        l_r.addWidget(self._spin_ramp_L0)
        l_r.addWidget(self._spin_ramp_L1)
        p_ramp_form.addRow("Target:", self._combo_ramp_target)
        p_ramp_form.addRow("Axis:", self._combo_ramp_axis)
        p_ramp_form.addRow("SF Min / Max:", sf_r)
        p_ramp_form.addRow("L Min / Max:", l_r)
        p_ramp_form.addRow("Transition Width:", self._spin_ramp_trans)
        self._grade_stack.addWidget(p_ramp)

        # Page 2: Multi-Knot Control Table
        p_mk = QWidget()
        p_mk_v = QVBoxLayout(p_mk)
        p_mk_v.setContentsMargins(0, 4, 0, 0)
        mk_form = QFormLayout()
        self._combo_mk_engine = QComboBox()
        self._combo_mk_engine.addItems(list(MULTI_KNOT_ENGINES))
        self._combo_mk_axis = QComboBox()
        self._combo_mk_axis.addItems(list(COORDINATE_AXES))
        self._combo_mk_kernel = QComboBox()
        self._combo_mk_kernel.addItems(list(BLENDING_KERNELS))
        for c in (self._combo_mk_engine, self._combo_mk_axis, self._combo_mk_kernel):
            c.currentIndexChanged.connect(self._on_generic_param_changed)
        mk_form.addRow("Blend Engine:", self._combo_mk_engine)
        mk_form.addRow("Coord Axis:", self._combo_mk_axis)
        mk_form.addRow("Kernel:", self._combo_mk_kernel)
        p_mk_v.addLayout(mk_form)

        self._tbl_knots = QTableWidget(0, 4)
        self._tbl_knots.setHorizontalHeaderLabels(["Pos (mm)", "L (mm)", "w (mm)", "SF"])
        self._tbl_knots.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._tbl_knots.setMinimumHeight(115)
        self._tbl_knots.itemChanged.connect(self._on_knot_table_edited)
        p_mk_v.addWidget(self._tbl_knots)

        mk_btns = QHBoxLayout()
        btn_add_k = QPushButton("+ Add Knot")
        btn_del_k = QPushButton("- Remove Knot")
        btn_add_k.clicked.connect(self._add_knot_row)
        btn_del_k.clicked.connect(self._remove_knot_row)
        mk_btns.addWidget(btn_add_k)
        mk_btns.addWidget(btn_del_k)
        p_mk_v.addLayout(mk_btns)
        self._grade_stack.addWidget(p_mk)

        # Page 3: Piecewise Discrete Bands
        p_pw = QWidget()
        p_pw_v = QVBoxLayout(p_pw)
        p_pw_v.setContentsMargins(0, 4, 0, 0)
        self._chk_pw_cont = QCheckBox("Phase-Continuous W Across Z-Breaks")
        self._chk_pw_cont.toggled.connect(self._on_generic_param_changed)
        p_pw_v.addWidget(self._chk_pw_cont)

        self._tbl_bands = QTableWidget(0, 4)
        self._tbl_bands.setHorizontalHeaderLabels(["Z Break", "L (mm)", "w (mm)", "SF"])
        self._tbl_bands.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._tbl_bands.setMinimumHeight(115)
        self._tbl_bands.itemChanged.connect(self._on_band_table_edited)
        p_pw_v.addWidget(self._tbl_bands)

        pw_btns = QHBoxLayout()
        btn_add_b = QPushButton("+ Add Zone")
        btn_del_b = QPushButton("- Remove Zone")
        btn_add_b.clicked.connect(self._add_band_row)
        btn_del_b.clicked.connect(self._remove_band_row)
        pw_btns.addWidget(btn_add_b)
        pw_btns.addWidget(btn_del_b)
        p_pw_v.addLayout(pw_btns)
        self._grade_stack.addWidget(p_pw)

        grade_v.addWidget(self._grade_stack)
        layout.addWidget(grp_grade)

        # 5. Surface Micro-Features
        grp_micro = self._make_help_group("micro_features", "5. Surface Micro-Features (Optional)")
        micro_v = QVBoxLayout(grp_micro)

        self._chk_tex = QCheckBox("Enable Procedural Surface Texture")
        self._chk_tex.toggled.connect(self._on_generic_param_changed)
        micro_v.addWidget(self._chk_tex)

        tex_form = QFormLayout()
        self._combo_tex_type = QComboBox()
        self._combo_tex_type.addItems(list(TEXTURE_TYPES))
        self._spin_tex_amp = self._make_spin(0.002, 0.5, 0.005, 3, " mm")
        self._spin_tex_wave = self._make_spin(0.01, 2.0, 0.01, 3, " mm")
        self._combo_tex_prof = QComboBox()
        self._combo_tex_prof.addItems(list(TEXTURE_PROFILES))
        self._combo_tex_disp = QComboBox()
        self._combo_tex_disp.addItems(list(TEXTURE_DISPLACEMENT_MODES))
        for w in (
            self._combo_tex_type,
            self._spin_tex_amp,
            self._spin_tex_wave,
            self._combo_tex_prof,
            self._combo_tex_disp,
        ):
            sig = w.currentIndexChanged if isinstance(w, QComboBox) else w.valueChanged
            sig.connect(self._on_generic_param_changed)
        tex_form.addRow("Texture:", self._combo_tex_type)
        tex_form.addRow("Amplitude / Pitch:", self._spin_tex_amp)
        tex_form.addRow("Wavelength:", self._spin_tex_wave)
        tex_form.addRow("Profile / Mode:", self._combo_tex_prof)
        tex_form.addRow("Displacement:", self._combo_tex_disp)
        micro_v.addLayout(tex_form)

        self._chk_pillars = QCheckBox("Enable Explicit Micropillar Forest")
        self._chk_pillars.toggled.connect(self._on_generic_param_changed)
        micro_v.addWidget(self._chk_pillars)

        pil_form = QFormLayout()
        self._spin_pil_d = self._make_spin(5.0, 500.0, 5.0, 1, " um")
        self._spin_pil_h = self._make_spin(10.0, 2000.0, 10.0, 1, " um")
        self._spin_pil_s = self._make_spin(20.0, 2000.0, 10.0, 1, " um")
        self._combo_pil_dist = QComboBox()
        self._combo_pil_dist.addItems(list(PILLAR_DISTRIBUTIONS))
        self._combo_pil_loc = QComboBox()
        self._combo_pil_loc.addItems(list(PILLAR_LOCATIONS))
        self._chk_pil_print = QCheckBox("Filter Printable Overhangs")
        for w in (
            self._spin_pil_d,
            self._spin_pil_h,
            self._spin_pil_s,
            self._combo_pil_dist,
            self._combo_pil_loc,
            self._chk_pil_print,
        ):
            if isinstance(w, QComboBox):
                w.currentIndexChanged.connect(self._on_generic_param_changed)
            elif isinstance(w, QCheckBox):
                w.toggled.connect(self._on_generic_param_changed)
            else:
                w.valueChanged.connect(self._on_generic_param_changed)
        pil_form.addRow("Diameter / Height:", self._spin_pil_d)
        pil_form.addRow("Pillar Height:", self._spin_pil_h)
        pil_form.addRow("Spacing:", self._spin_pil_s)
        pil_form.addRow("Distribution:", self._combo_pil_dist)
        pil_form.addRow("Scope:", self._combo_pil_loc)
        pil_form.addRow("", self._chk_pil_print)
        micro_v.addLayout(pil_form)

        layout.addWidget(grp_micro)
        layout.addStretch(1)

        scroll.setWidget(body)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    # ------------------------------------------------------------------
    # Right Diagnostics, Calibration & Export Dock
    # ------------------------------------------------------------------
    def _build_right_dock(self) -> None:
        dock = QDockWidget("Diagnostics, Solver && Export", self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock.setMinimumWidth(380)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(6)

        # 1. Live Telemetry & Voxel Budget
        grp_tel = self._make_help_group("telemetry", "Field && Voxel Telemetry")
        tel_v = QVBoxLayout(grp_tel)
        self._lbl_telemetry = QLabel("")
        self._lbl_telemetry.setStyleSheet("font-family: Consolas, monospace; color: #cbd5e1;")
        self._lbl_telemetry.setWordWrap(True)
        tel_v.addWidget(self._lbl_telemetry)
        layout.addWidget(grp_tel)

        # 2. Calibration & MIS Status
        grp_cal = self._make_help_group("calibration_status", "Calibration && Pore Check")
        cal_v = QVBoxLayout(grp_cal)
        self._lbl_cal_status = QLabel("Analytical LUT ready. Click 'Calibrate Exact' for Newton MIS convergence.")
        self._lbl_cal_status.setWordWrap(True)
        self._lbl_cal_status.setStyleSheet("color: #94a3b8;")
        cal_v.addWidget(self._lbl_cal_status)
        self._chk_post_mis = QCheckBox("Verify MIS Pore Size on Exported Mesh")
        self._chk_post_mis.toggled.connect(self._on_generic_param_changed)
        cal_v.addWidget(self._chk_post_mis)
        layout.addWidget(grp_cal)

        # 3. Production Execution
        grp_run = self._make_help_group("production_synthesis", "Production Synthesis")
        run_v = QVBoxLayout(grp_run)
        self._btn_generate = QPushButton("Generate Production Lattice")
        self._btn_generate.setObjectName("primaryBtn")
        self._btn_generate.clicked.connect(self._on_generate_clicked)
        run_v.addWidget(self._btn_generate)

        btn_preview_now = QPushButton("Refresh 3D Footprint && Slice")
        btn_preview_now.clicked.connect(self.refresh_visual_previews)
        run_v.addWidget(btn_preview_now)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        run_v.addWidget(self._progress)

        self._log_box = QPlainTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMaximumHeight(150)
        self._log_box.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        run_v.addWidget(self._log_box)
        layout.addWidget(grp_run)

        # 4. Export & Reproducibility Ledger
        grp_exp = self._make_help_group("export_ledger", "Export && Recipe Ledger")
        exp_v = QVBoxLayout(grp_exp)

        fmt_row = QHBoxLayout()
        self._chk_exp_stl = QCheckBox(".STL")
        self._has_3mf = has_threemf_support()
        self._chk_exp_3mf = QCheckBox(".3MF" if self._has_3mf else ".3MF (needs lxml)")
        if not self._has_3mf:
            self._chk_exp_3mf.setEnabled(False)
            self._chk_exp_3mf.setToolTip("Requires optional 'lxml' package in the active Python environment.")
        self._chk_exp_manifest = QCheckBox("JSON && Manifest")
        for c in (self._chk_exp_stl, self._chk_exp_3mf, self._chk_exp_manifest):
            c.toggled.connect(self._on_generic_param_changed)
            fmt_row.addWidget(c)
        exp_v.addLayout(fmt_row)

        self._edit_custom_name = QLineEdit()
        self._edit_custom_name.setPlaceholderText("Optional custom output stem...")
        self._edit_custom_name.textChanged.connect(self._on_generic_param_changed)
        exp_v.addWidget(self._edit_custom_name)

        self._chk_show_gen_mesh_dock = QCheckBox("Load Generated 3D Mesh in Viewport (Off by default)")
        self._chk_show_gen_mesh_dock.toggled.connect(self._on_show_generated_toggled)
        exp_v.addWidget(self._chk_show_gen_mesh_dock)

        recipe_row = QHBoxLayout()
        btn_save_r = QPushButton("Save Recipe...")
        btn_load_r = QPushButton("Load Recipe...")
        btn_batch_r = QPushButton("Run Batch Queue...")
        btn_save_r.clicked.connect(self._on_save_recipe)
        btn_load_r.clicked.connect(self._on_load_recipe)
        btn_batch_r.clicked.connect(self._on_batch_recipes_clicked)
        recipe_row.addWidget(btn_save_r)
        recipe_row.addWidget(btn_load_r)
        exp_v.addLayout(recipe_row)
        exp_v.addWidget(btn_batch_r)

        out_row = QHBoxLayout()
        btn_open_out = QPushButton("Open Output Folder")
        self._btn_save_copy = QPushButton("Save Copy As...")
        self._btn_save_copy.setEnabled(False)
        btn_open_out.clicked.connect(self._on_open_output_folder)
        self._btn_save_copy.clicked.connect(self._on_save_copy_as)
        out_row.addWidget(btn_open_out)
        out_row.addWidget(self._btn_save_copy)
        exp_v.addLayout(out_row)

        self._lbl_export_summary = QLabel("No production mesh generated yet.")
        self._lbl_export_summary.setWordWrap(True)
        self._lbl_export_summary.setStyleSheet("font-family: Consolas, monospace; color: #94a3b8;")
        exp_v.addWidget(self._lbl_export_summary)
        layout.addWidget(grp_exp)

        layout.addStretch(1)
        scroll.setWidget(body)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _make_spin(
        self,
        vmin: float,
        vmax: float,
        step: float,
        decimals: int,
        suffix: str,
    ) -> _NoWheelDoubleSpinBox:
        sp = _NoWheelDoubleSpinBox()
        sp.setRange(vmin, vmax)
        sp.setSingleStep(step)
        sp.setDecimals(decimals)
        sp.setMinimumWidth(84)
        if suffix:
            sp.setSuffix(suffix)
        return sp

    def _bind_shortcuts(self) -> None:
        for key, preset in (("1", "iso"), ("2", "top"), ("3", "front"), ("4", "side"), ("R", "reset")):
            act = QAction(self)
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(lambda _c=False, p=preset: self._set_camera(p))
            self.addAction(act)
        act_snap = QAction(self)
        act_snap.setShortcut(QKeySequence("Ctrl+P"))
        act_snap.triggered.connect(self._on_snapshot_clicked)
        self.addAction(act_snap)

    # ------------------------------------------------------------------
    # State <-> UI Synchronization
    # ------------------------------------------------------------------
    @staticmethod
    def _set_combo_by_data(combo: QComboBox, data_val: str) -> None:
        idx = combo.findData(str(data_val or ""))
        combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _sync_widgets_from_state(self) -> None:
        self._updating_ui = True
        try:
            s = self.state
            s.exact_cad_trim = True
            if s.preview_mode == "3D Flying Edges (Fine)":
                s.preview_mode = "3D Flying Edges (Coarse)"

            self._rb_prim.setChecked(s.geometry_source == "Primitive")
            self._rb_file.setChecked(s.geometry_source != "Primitive")
            self._cad_stack.setCurrentIndex(0 if s.geometry_source == "Primitive" else 1)

            self._combo_prim_shape.setCurrentText(s.primitive_shape)
            self._spin_prim_x.setValue(float(s.prim_x_mm))
            self._spin_prim_y.setValue(float(s.prim_y_mm))
            self._spin_prim_z.setValue(float(s.prim_z_mm))
            self._edit_cad_path.setText(s.cad_file_path)
            self._chk_center_origin.setChecked(bool(s.center_origin))

            self._combo_arch.setCurrentText(s.lattice_type)
            self._chk_true_woodpile.setChecked(bool(s.woodpile_true_mode))
            self._chk_true_woodpile.setVisible(s.lattice_type == "Woodpile")
            has_skin = s.export_mode in ("skin", "combined")
            self._chk_generate_skin.setChecked(has_skin)
            self._skin_options_widget.setVisible(has_skin)
            self._spin_shell_t.setValue(float(s.shell_thickness_mm))
            self._spin_blend_r.setValue(float(s.blend_radius_mm))
            self._combo_blend_method.setCurrentText(s.blend_method)

            self._set_combo_by_data(self._combo_size_drv, s.size_driver)
            self._set_combo_by_data(self._combo_density_drv, s.density_driver)
            self._spin_L.setValue(float(s.unit_cell_size_mm))
            self._spin_pore.setValue(float(s.pore_size_mm))
            self._spin_sf.setValue(float(s.solid_fraction))
            self._spin_wall.setValue(float(s.wall_thickness_mm))
            self._update_four_way_box_states()
            self._chk_auto_res.setChecked(bool(s.auto_resolution))
            self._spin_res.setEnabled(not s.auto_resolution)
            if s.auto_resolution:
                prof = evaluate_grading_profile_1d(s, self._cad_mesh, n_samples=40)
                min_w = float(np.min(prof["w_mm"])) if len(prof["w_mm"]) else s.wall_thickness_mm
                min_L = float(np.min(prof["L_mm"])) if len(prof["L_mm"]) else s.unit_cell_size_mm
                s.resolution_mm = recommended_nyquist_resolution_mm(min_w, min_L)
            self._spin_res.setValue(float(s.resolution_mm))

            self._combo_grading_mode.setCurrentText(s.grading_mode)
            self._grade_stack.setCurrentIndex(GRADING_MODES.index(s.grading_mode))

            self._spin_rx.setValue(float(s.rotation_x_deg))
            self._spin_ry.setValue(float(s.rotation_y_deg))
            self._spin_rz.setValue(float(s.rotation_z_deg))

            self._combo_ramp_target.setCurrentText(s.ramp_target)
            self._combo_ramp_axis.setCurrentText(s.ramp_axis)
            self._spin_ramp_sf0.setValue(float(s.ramp_sf_min))
            self._spin_ramp_sf1.setValue(float(s.ramp_sf_max))
            self._spin_ramp_L0.setValue(float(s.ramp_L_min_mm))
            self._spin_ramp_L1.setValue(float(s.ramp_L_max_mm))
            self._spin_ramp_trans.setValue(float(s.ramp_transition_width_mm))

            self._combo_mk_engine.setCurrentText(s.multi_knot_engine)
            self._combo_mk_axis.setCurrentText(s.multi_knot_axis)
            self._combo_mk_kernel.setCurrentText(s.multi_knot_kernel)
            self._populate_knot_table()

            self._chk_pw_cont.setChecked(bool(s.piecewise_phase_continuous))
            self._populate_band_table()

            self._chk_tex.setChecked(bool(s.enable_surface_texture))
            self._combo_tex_type.setCurrentText(s.texture_type)
            self._spin_tex_amp.setValue(float(s.texture_amplitude_mm))
            self._spin_tex_wave.setValue(float(s.texture_wavelength_mm))
            self._combo_tex_prof.setCurrentText(s.texture_profile)
            self._combo_tex_disp.setCurrentText(s.texture_displacement_mode)

            self._chk_pillars.setChecked(bool(s.enable_micropillars))
            self._spin_pil_d.setValue(float(s.pillar_diameter_um))
            self._spin_pil_h.setValue(float(s.pillar_height_um))
            self._spin_pil_s.setValue(float(s.pillar_spacing_um))
            self._combo_pil_dist.setCurrentText(s.pillar_distribution)
            self._combo_pil_loc.setCurrentText(s.pillar_location)
            self._chk_pil_print.setChecked(bool(s.pillar_filter_printable))

            self._chk_post_mis.setChecked(bool(s.run_post_mis_check))
            self._chk_exp_stl.setChecked(bool(s.export_stl))
            if not self._has_3mf:
                s.export_3mf = False
            self._chk_exp_3mf.setChecked(bool(s.export_3mf) and self._has_3mf)
            self._chk_exp_manifest.setChecked(bool(s.write_manifest))
            self._edit_custom_name.setText(s.custom_output_name)
            self._combo_preview_mode.setCurrentText(s.preview_mode)
            self._chk_show_gen_mesh_tb.setChecked(bool(s.show_generated_3d_mesh))
            self._chk_show_gen_mesh_dock.setChecked(bool(s.show_generated_3d_mesh))
        finally:
            self._updating_ui = False

    def _populate_knot_table(self) -> None:
        self._tbl_knots.blockSignals(True)
        knots = self.state.multi_knot_knots
        self._tbl_knots.setRowCount(len(knots))
        for r, k in enumerate(knots):
            for c, key in enumerate(("pos_mm", "L_mm", "w_mm", "sf")):
                self._tbl_knots.setItem(r, c, QTableWidgetItem(f"{float(k[key]):.3f}"))
        self._tbl_knots.blockSignals(False)

    def _populate_band_table(self) -> None:
        self._tbl_bands.blockSignals(True)
        n_zones = max(len(self.state.piecewise_L_mm), 1)
        self._tbl_bands.setRowCount(n_zones)
        breaks = self.state.piecewise_breaks_mm
        for r in range(n_zones):
            b_txt = f"{float(breaks[r]):.3f}" if r < len(breaks) else "Top"
            self._tbl_bands.setItem(r, 0, QTableWidgetItem(b_txt))
            self._tbl_bands.setItem(r, 1, QTableWidgetItem(f"{float(self.state.piecewise_L_mm[r]):.3f}"))
            self._tbl_bands.setItem(r, 2, QTableWidgetItem(f"{float(self.state.piecewise_w_mm[r]):.3f}"))
            self._tbl_bands.setItem(r, 3, QTableWidgetItem(f"{float(self.state.piecewise_sf[r]):.3f}"))
        self._tbl_bands.blockSignals(False)

    def _pull_state_from_widgets(self) -> None:
        s = self.state
        s.geometry_source = "Primitive" if self._rb_prim.isChecked() else "File"
        s.primitive_shape = self._combo_prim_shape.currentText()
        s.prim_x_mm = self._spin_prim_x.value()
        s.prim_y_mm = self._spin_prim_y.value()
        s.prim_z_mm = self._spin_prim_z.value()
        s.cad_file_path = self._edit_cad_path.text().strip()
        s.center_origin = self._chk_center_origin.isChecked()

        s.lattice_type = self._combo_arch.currentText()
        s.woodpile_true_mode = self._chk_true_woodpile.isChecked()
        s.export_mode = "combined" if self._chk_generate_skin.isChecked() else "core"
        s.shell_thickness_mm = self._spin_shell_t.value()
        s.blend_radius_mm = self._spin_blend_r.value()
        s.blend_method = self._combo_blend_method.currentText()
        s.exact_cad_trim = True

        s.size_driver = str(self._combo_size_drv.currentData() or "")
        s.density_driver = str(self._combo_density_drv.currentData() or "")
        s.unit_cell_size_mm = self._spin_L.value()
        s.pore_size_mm = self._spin_pore.value()
        s.solid_fraction = self._spin_sf.value()
        s.wall_thickness_mm = self._spin_wall.value()
        s.auto_resolution = self._chk_auto_res.isChecked()
        s.grading_mode = self._combo_grading_mode.currentText()
        if s.auto_resolution:
            prof = evaluate_grading_profile_1d(s, self._cad_mesh, n_samples=40)
            min_w = float(np.min(prof["w_mm"])) if len(prof["w_mm"]) else s.wall_thickness_mm
            min_L = float(np.min(prof["L_mm"])) if len(prof["L_mm"]) else s.unit_cell_size_mm
            s.resolution_mm = recommended_nyquist_resolution_mm(min_w, min_L)
        else:
            s.resolution_mm = self._spin_res.value()
        s.rotation_x_deg = self._spin_rx.value()
        s.rotation_y_deg = self._spin_ry.value()
        s.rotation_z_deg = self._spin_rz.value()

        s.ramp_target = self._combo_ramp_target.currentText()
        s.ramp_axis = self._combo_ramp_axis.currentText()
        s.ramp_sf_min = self._spin_ramp_sf0.value()
        s.ramp_sf_max = self._spin_ramp_sf1.value()
        s.ramp_L_min_mm = self._spin_ramp_L0.value()
        s.ramp_L_max_mm = self._spin_ramp_L1.value()
        s.ramp_transition_width_mm = self._spin_ramp_trans.value()

        s.multi_knot_engine = self._combo_mk_engine.currentText()
        s.multi_knot_axis = self._combo_mk_axis.currentText()
        s.multi_knot_kernel = self._combo_mk_kernel.currentText()
        s.piecewise_phase_continuous = self._chk_pw_cont.isChecked()

        s.enable_surface_texture = self._chk_tex.isChecked()
        s.texture_type = self._combo_tex_type.currentText()
        s.texture_amplitude_mm = self._spin_tex_amp.value()
        s.texture_wavelength_mm = self._spin_tex_wave.value()
        s.texture_profile = self._combo_tex_prof.currentText()
        s.texture_displacement_mode = self._combo_tex_disp.currentText()

        s.enable_micropillars = self._chk_pillars.isChecked()
        s.pillar_diameter_um = self._spin_pil_d.value()
        s.pillar_height_um = self._spin_pil_h.value()
        s.pillar_spacing_um = self._spin_pil_s.value()
        s.pillar_distribution = self._combo_pil_dist.currentText()
        s.pillar_location = self._combo_pil_loc.currentText()
        s.pillar_filter_printable = self._chk_pil_print.isChecked()

        s.run_post_mis_check = self._chk_post_mis.isChecked()
        s.export_stl = self._chk_exp_stl.isChecked()
        s.export_3mf = self._chk_exp_3mf.isChecked() and self._has_3mf
        s.write_manifest = self._chk_exp_manifest.isChecked()
        s.custom_output_name = self._edit_custom_name.text().strip()
        s.preview_mode = self._combo_preview_mode.currentText()

    # ------------------------------------------------------------------
    # Event Handlers & Debounced Preview
    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        self._log_box.appendPlainText(msg)

    def _on_cad_source_changed(self) -> None:
        if self._updating_ui:
            return
        self._cad_stack.setCurrentIndex(0 if self._rb_prim.isChecked() else 1)
        if self._rb_file.isChecked() and not self._edit_cad_path.text().strip():
            if self._combo_fixtures.count() > 0:
                self._edit_cad_path.setText(str(self._combo_fixtures.currentData()))
        self._pull_state_from_widgets()
        self._rebuild_cad_and_refresh()

    def _on_fixture_selected(self) -> None:
        if self._updating_ui or self._combo_fixtures.count() == 0:
            return
        self._edit_cad_path.setText(str(self._combo_fixtures.currentData()))
        self._pull_state_from_widgets()
        self._rebuild_cad_and_refresh()

    def _browse_cad_stl(self) -> None:
        start_dir = str(Path(self.state.cad_file_path).parent) if self.state.cad_file_path else "test_parts"
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CAD STL Mesh", start_dir, "STL Meshes (*.stl);;All Files (*)"
        )
        if path:
            self._rb_file.setChecked(True)
            self._edit_cad_path.setText(path)
            self._pull_state_from_widgets()
            self._rebuild_cad_and_refresh()

    def _on_cad_param_changed(self) -> None:
        if self._updating_ui:
            return
        self._pull_state_from_widgets()
        self._rebuild_cad_and_refresh()

    def _on_repair_cad_clicked(self) -> None:
        if self._cad_mesh is None:
            return
        repaired, health = repair_cad_mesh_in_memory(self._cad_mesh)
        self._cad_mesh = repaired
        self._cad_health = health
        self._update_cad_health_badge()
        self._log(
            f"CAD gentle repair complete: watertight={health['is_watertight']}, "
            f"faces={health['face_count']:,}"
        )
        self.refresh_visual_previews(reset_cam=False)

    def _on_generate_skin_toggled(self, checked: bool) -> None:
        self._skin_options_widget.setVisible(bool(checked))
        if self._updating_ui:
            return
        self._on_generic_param_changed()

    def _on_arch_changed(self) -> None:
        if self._updating_ui:
            return
        self._chk_true_woodpile.setVisible(self._combo_arch.currentText() == "Woodpile")
        d1 = str(self._combo_size_drv.currentData() or "")
        d2 = str(self._combo_density_drv.currentData() or "")
        if d1 and d2 and d1 != d2:
            self._on_four_way_changed(d1)
        else:
            self._on_generic_param_changed()

    def _on_drivers_changed(self, changed_slot: int) -> None:
        if self._updating_ui:
            return
        d1 = str(self._combo_size_drv.currentData() or "")
        d2 = str(self._combo_density_drv.currentData() or "")
        if d1 and d2 and d1 == d2:
            other = self._combo_density_drv if changed_slot == 1 else self._combo_size_drv
            other.blockSignals(True)
            other.setCurrentIndex(0)
            other.blockSignals(False)

        was_updating = self._updating_ui
        self._updating_ui = True
        try:
            self._pull_state_from_widgets()
            self._update_four_way_box_states()
        finally:
            self._updating_ui = was_updating
        self._update_telemetry_label()
        self._preview_timer.start()

    def _update_four_way_box_states(self) -> None:
        d1 = str(self._combo_size_drv.currentData() or "")
        d2 = str(self._combo_density_drv.currentData() or "")
        ready = bool(d1 and d2 and d1 != d2)
        spin_map = (
            ("unit_cell", self._spin_L),
            ("pore_size", self._spin_pore),
            ("solid_fraction", self._spin_sf),
            ("wall_thickness", self._spin_wall),
        )
        for key, spin in spin_map:
            spin.setEnabled(ready and key in (d1, d2))

        if ready:
            metrics = solve_four_way_pair(
                self.state.lattice_type,
                d1,
                d2,
                unit_cell_size_mm=self._spin_L.value(),
                pore_size_mm=self._spin_pore.value(),
                solid_fraction=self._spin_sf.value(),
                wall_thickness_mm=self._spin_wall.value(),
            )
            self._apply_solved_four_way_metrics(metrics)

    def _apply_solved_four_way_metrics(self, metrics: dict[str, float]) -> None:
        self.state.unit_cell_size_mm = metrics["unit_cell_size_mm"]
        self.state.pore_size_mm = metrics["pore_size_mm"]
        self.state.solid_fraction = metrics["solid_fraction"]
        self.state.wall_thickness_mm = metrics["wall_thickness_mm"]

        for spin, val in (
            (self._spin_L, metrics["unit_cell_size_mm"]),
            (self._spin_pore, metrics["pore_size_mm"]),
            (self._spin_sf, metrics["solid_fraction"]),
            (self._spin_wall, metrics["wall_thickness_mm"]),
        ):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

        if self.state.auto_resolution:
            self.state.resolution_mm = metrics["recommended_resolution_mm"]
            self._spin_res.blockSignals(True)
            self._spin_res.setValue(metrics["recommended_resolution_mm"])
            self._spin_res.blockSignals(False)

    def _on_four_way_changed(self, _driven_by: str) -> None:
        if self._updating_ui:
            return
        d1 = str(self._combo_size_drv.currentData() or "")
        d2 = str(self._combo_density_drv.currentData() or "")
        if not (d1 and d2 and d1 != d2):
            return
        self._updating_ui = True
        try:
            self._pull_state_from_widgets()
            metrics = solve_four_way_pair(
                self.state.lattice_type,
                d1,
                d2,
                unit_cell_size_mm=self.state.unit_cell_size_mm,
                pore_size_mm=self.state.pore_size_mm,
                solid_fraction=self.state.solid_fraction,
                wall_thickness_mm=self.state.wall_thickness_mm,
            )
            self._apply_solved_four_way_metrics(metrics)
        finally:
            self._updating_ui = False
        self._update_telemetry_label()
        self._preview_timer.start()

    def _on_auto_res_toggled(self) -> None:
        if self._updating_ui:
            return
        auto = self._chk_auto_res.isChecked()
        self._spin_res.setEnabled(not auto)
        if auto:
            self._updating_ui = True
            self._spin_res.setValue(
                recommended_nyquist_resolution_mm(
                    self._spin_wall.value(), self._spin_L.value()
                )
            )
            self._updating_ui = False
        self._on_generic_param_changed()

    def _on_grading_mode_changed(self) -> None:
        if self._updating_ui:
            return
        idx = self._combo_grading_mode.currentIndex()
        self._grade_stack.setCurrentIndex(idx)
        self._on_generic_param_changed()

    def _on_preset_selected(self, idx: int) -> None:
        if self._updating_ui or idx <= 0:
            return
        preset_name = self._combo_preset.itemText(idx)
        bounds = self._cad_mesh.bounds if self._cad_mesh is not None else None
        self.state.apply_preset(preset_name, bounds=bounds)
        self._sync_widgets_from_state()
        self._combo_preset.setCurrentIndex(0)
        self._update_telemetry_label()
        self._preview_timer.start()
        self._log(f"Applied preset: {preset_name}")

    def _on_knot_table_edited(self, _item: QTableWidgetItem) -> None:
        if self._updating_ui:
            return
        knots: list[dict[str, float]] = []
        for r in range(self._tbl_knots.rowCount()):
            try:
                knots.append(
                    {
                        "pos_mm": float(self._tbl_knots.item(r, 0).text()),
                        "L_mm": max(float(self._tbl_knots.item(r, 1).text()), 0.2),
                        "w_mm": max(float(self._tbl_knots.item(r, 2).text()), 0.03),
                        "sf": float(np.clip(float(self._tbl_knots.item(r, 3).text()), 0.05, 0.90)),
                    }
                )
            except Exception:
                return
        self.state.multi_knot_knots = knots
        self._update_telemetry_label()
        self._preview_timer.start()

    def _add_knot_row(self) -> None:
        last = self.state.multi_knot_knots[-1] if self.state.multi_knot_knots else {"pos_mm": 0.0, "L_mm": 4.0, "w_mm": 0.5, "sf": 0.35}
        self.state.multi_knot_knots.append(
            {
                "pos_mm": round(float(last["pos_mm"]) + 2.5, 3),
                "L_mm": float(last["L_mm"]),
                "w_mm": float(last["w_mm"]),
                "sf": float(last["sf"]),
            }
        )
        self._populate_knot_table()
        self._preview_timer.start()

    def _remove_knot_row(self) -> None:
        if len(self.state.multi_knot_knots) <= 2:
            return
        row = self._tbl_knots.currentRow()
        idx = row if 0 <= row < len(self.state.multi_knot_knots) else -1
        self.state.multi_knot_knots.pop(idx)
        self._populate_knot_table()
        self._preview_timer.start()

    def _on_band_table_edited(self, _item: QTableWidgetItem) -> None:
        if self._updating_ui:
            return
        n_rows = self._tbl_bands.rowCount()
        breaks: list[float] = []
        Ls: list[float] = []
        ws: list[float] = []
        sfs: list[float] = []
        try:
            for r in range(n_rows):
                if r < n_rows - 1:
                    breaks.append(float(self._tbl_bands.item(r, 0).text()))
                Ls.append(max(float(self._tbl_bands.item(r, 1).text()), 0.2))
                ws.append(max(float(self._tbl_bands.item(r, 2).text()), 0.03))
                sfs.append(float(np.clip(float(self._tbl_bands.item(r, 3).text()), 0.05, 0.90)))
        except Exception:
            return
        self.state.piecewise_breaks_mm = breaks
        self.state.piecewise_L_mm = Ls
        self.state.piecewise_w_mm = ws
        self.state.piecewise_sf = sfs
        self._update_telemetry_label()
        self._preview_timer.start()

    def _add_band_row(self) -> None:
        last_b = self.state.piecewise_breaks_mm[-1] if self.state.piecewise_breaks_mm else 0.0
        self.state.piecewise_breaks_mm.append(round(float(last_b) + 2.5, 3))
        self.state.piecewise_L_mm.append(float(self.state.piecewise_L_mm[-1]))
        self.state.piecewise_w_mm.append(float(self.state.piecewise_w_mm[-1]))
        self.state.piecewise_sf.append(float(self.state.piecewise_sf[-1]))
        self._populate_band_table()
        self._preview_timer.start()

    def _remove_band_row(self) -> None:
        if len(self.state.piecewise_L_mm) <= 2:
            return
        self.state.piecewise_breaks_mm.pop()
        self.state.piecewise_L_mm.pop()
        self.state.piecewise_w_mm.pop()
        self.state.piecewise_sf.pop()
        self._populate_band_table()
        self._preview_timer.start()

    def _on_generic_param_changed(self) -> None:
        if self._updating_ui:
            return
        self._pull_state_from_widgets()
        self._update_telemetry_label()
        self._preview_timer.start()

    # ------------------------------------------------------------------
    # CAD + Visual Preview Updates
    # ------------------------------------------------------------------
    def _rebuild_cad_and_refresh(self) -> None:
        try:
            mesh, health = build_or_load_cad_mesh(self.state)
            self._cad_mesh = mesh
            self._cad_health = health
            self._update_cad_health_badge()
            if self._scene is not None:
                self._scene.set_cad_mesh(mesh, reset_cam=True)
            self._update_telemetry_label()
            self.refresh_visual_previews(reset_cam=True)
        except Exception as exc:
            self._lbl_cad_health.setText(f"CAD Error: {exc}")
            self._lbl_cad_health.setStyleSheet("color: #f87171;")

    def _update_cad_health_badge(self) -> None:
        h = self._cad_health
        if not h:
            return
        wt = h.get("is_watertight", False)
        ext = h.get("extents_mm", [0.0, 0.0, 0.0])
        faces = h.get("face_count", 0)
        if wt:
            self._lbl_cad_health.setText(
                f"Watertight OK | {faces:,} faces | {ext[0]:.1f}x{ext[1]:.1f}x{ext[2]:.1f} mm"
            )
            self._lbl_cad_health.setStyleSheet("color: #34d399; font-weight: 600;")
        else:
            self._lbl_cad_health.setText(
                f"Non-Watertight (bnd={h.get('boundary_edges', 0)}) | Click Repair"
            )
            self._lbl_cad_health.setStyleSheet("color: #fbbf24; font-weight: 600;")

    def _update_telemetry_label(self) -> None:
        bud = estimate_voxel_budget(self._cad_mesh, self.state.resolution_mm)
        prof = evaluate_grading_profile_1d(self.state, self._cad_mesh, n_samples=25)
        L_min, L_max = float(np.min(prof["L_mm"])), float(np.max(prof["L_mm"]))
        sf_min, sf_max = float(np.min(prof["sf"])), float(np.max(prof["sf"]))
        lines = [
            f"Grid: {bud['nx']} x {bud['ny']} x {bud['nz']} ({bud['voxels']:,} vox)",
            f"Est. Memory: {bud['est_ram_mb']:.1f} MB | Pitch: {self.state.resolution_mm:.3f} mm",
            f"Profile L: {L_min:.2f} .. {L_max:.2f} mm | SF: {sf_min:.2f} .. {sf_max:.2f}",
        ]
        if bud["warning"]:
            lines.append(f"WARNING: {bud['warning']}")
        self._lbl_telemetry.setText("\n".join(lines))

    def _format_viewport_hud(self) -> str:
        if self._cad_mesh is None or self._cad_mesh.bounds is None:
            return ""
        ext = np.asarray(self._cad_mesh.bounds[1] - self._cad_mesh.bounds[0], float)
        flags: list[str] = [
            self.state.preview_mode.split(" (")[0],
            self.state.lattice_type,
            f"L={self.state.unit_cell_size_mm:.2f}mm",
            f"SF={self.state.solid_fraction:.2f}",
        ]
        if self.state.enable_surface_texture:
            flags.append(f"Tex:{self.state.texture_type}")
        if self.state.enable_micropillars:
            flags.append(f"Pillars:{int(self.state.pillar_diameter_um)}um")
        if self._chk_cutaway.isChecked():
            flags.append(f"Cut:{self._combo_cut_axis.currentText()}@{int(self._slider_cut.value())}%")
        return (
            f"{ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm\n"
            + " | ".join(flags)
        )

    def refresh_visual_previews(self, *, reset_cam: bool = False) -> None:
        if self._cad_mesh is None:
            return
        if self._scene is not None and self._chk_footprint.isChecked():
            try:
                fp_poly = build_surface_footprint_polydata(self.state, self._cad_mesh)
                self._scene.set_surface_footprint(fp_poly)
                self._scene.set_cutaway_cap(self.state, self._cad_mesh)
                if self.state.enable_micropillars:
                    pil_poly = build_micropillar_preview_polydata(
                        self.state,
                        self._cad_mesh,
                        footprint_poly=fp_poly,
                    )
                    self._scene.set_micropillar_preview(pil_poly)
                else:
                    self._scene.set_micropillar_preview(None)
                self._scene.update_corner_hud(self._format_viewport_hud())
                if reset_cam:
                    self._scene.set_camera_preset("iso")
                self._render_vtk()
            except Exception as exc:
                self._log(f"Preview update skipped: {exc}")
        elif self._scene is not None:
            self._scene.update_corner_hud(self._format_viewport_hud())
            self._render_vtk()
        if self._tabs.currentIndex() == 1:
            self._refresh_slice_tab()

    def _render_vtk(self) -> None:
        if self._vtk_widget is not None:
            self._vtk_widget.GetRenderWindow().Render()

    def _set_camera(self, preset: str) -> None:
        if self._scene is not None:
            self._scene.set_camera_preset(preset)
            self._render_vtk()

    def _on_preview_mode_changed(self, mode: str) -> None:
        if self._updating_ui:
            return
        self.state.preview_mode = mode
        if mode in ("3D Flying Edges (Coarse)", "3D Flying Edges (Fine)") and self._opacity_combo.currentIndex() == 0:
            self._opacity_combo.setCurrentIndex(1)  # 25% Ghost reveals 3D interior
        elif mode == "2D Surface (Ultra-Sharp)" and self._opacity_combo.currentIndex() in (1, 2):
            self._opacity_combo.setCurrentIndex(0)  # 100% Opaque Gray backdrop
        self.refresh_visual_previews()

    def _on_opacity_changed(self, idx: int) -> None:
        op_map = {0: 1.0, 1: 0.25, 2: 0.12, 3: 0.0}
        if self._scene is not None:
            self._scene.set_cad_opacity(op_map.get(idx, 1.0))
            self._render_vtk()

    def _on_footprint_toggled(self, checked: bool) -> None:
        if self._scene is not None:
            self._scene.set_footprint_visible(checked)
            if checked:
                self.refresh_visual_previews()
            else:
                self._render_vtk()

    def _on_show_generated_toggled(self, checked: bool) -> None:
        if self._updating_ui:
            return
        self._updating_ui = True
        try:
            self.state.show_generated_3d_mesh = bool(checked)
            self._chk_show_gen_mesh_tb.setChecked(bool(checked))
            self._chk_show_gen_mesh_dock.setChecked(bool(checked))
        finally:
            self._updating_ui = False

        if self._scene is not None:
            if checked and self._generated_mesh is not None:
                self._scene.set_generated_mesh(self._generated_mesh, visible=True)
            else:
                self._scene.set_generated_mesh_visible(checked)
            self._render_vtk()

    def _on_cutaway_slider_pressed(self) -> None:
        self._cut_dragging = True
        self._cutaway_cap_timer.stop()

    def _on_cutaway_slider_released(self) -> None:
        self._cut_dragging = False
        self._cutaway_cap_timer.stop()
        self._finalize_cutaway_cap()

    def _on_cutaway_value_changed(self, val: int) -> None:
        pct = float(val)
        self._lbl_cut_pct.setText(f"{int(pct)}%")
        if self._scene is not None:
            self._scene.set_cutaway(
                enabled=self._chk_cutaway.isChecked(),
                axis=self._combo_cut_axis.currentText(),
                pct=pct,
                interactive=True,
            )
            self._scene.update_corner_hud(self._format_viewport_hud())
            self._render_vtk()
        if not self._cut_dragging and self._chk_cutaway.isChecked():
            self._cutaway_cap_timer.start()

    def _finalize_cutaway_cap(self) -> None:
        pct = float(self._slider_cut.value())
        self._lbl_cut_pct.setText(f"{int(pct)}%")
        if self._scene is not None:
            self._scene.set_cutaway(
                enabled=self._chk_cutaway.isChecked(),
                axis=self._combo_cut_axis.currentText(),
                pct=pct,
                interactive=False,
            )
            self._scene.set_cutaway_cap(self.state, self._cad_mesh)
            self._scene.update_corner_hud(self._format_viewport_hud())
            self._render_vtk()

    def _on_cutaway_changed(self) -> None:
        self._cutaway_cap_timer.stop()
        self._finalize_cutaway_cap()

    def _on_slice_slider_changed(self, val: int) -> None:
        self._lbl_slice_pos.setText(f"{int(val)}%")
        if self._tabs.currentIndex() == 1:
            self._slice_timer.start()

    def _on_tab_changed(self, idx: int) -> None:
        if idx == 1:
            self._refresh_slice_tab()

    def _refresh_slice_tab(self) -> None:
        if self._cad_mesh is None:
            return
        pct = float(self._slider_slice_pos.value())
        self._lbl_slice_pos.setText(f"{int(pct)}%")
        px_text = self._combo_slice_px.currentText().split()[0]
        px_mm = float(px_text)
        res = compute_2d_slice_image(
            self.state,
            self._cad_mesh,
            axis=self._combo_slice_axis.currentText(),
            slice_pos_pct=pct,
            pixel_size_mm=px_mm,
        )
        rgb = np.ascontiguousarray(res.rgb_image)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            max(self._lbl_slice_image.width() - 12, 420),
            max(self._lbl_slice_image.height() - 12, 420),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )

        # Overlay calibrated physical scale bar on the display pixmap only
        if pix.width() > 80 and res.width_mm > 1e-6:
            px_per_mm = float(pix.width()) / float(res.width_mm)
            target_px = 0.22 * float(pix.width())
            candidates = (0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
            bar_mm = min(candidates, key=lambda c: abs(c * px_per_mm - target_px))
            bar_px = max(int(round(bar_mm * px_per_mm)), 18)
            bar_px = min(bar_px, pix.width() - 32)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing, True)
            margin_x, margin_y = 14, pix.height() - 18
            bg_rect = QRectF(margin_x - 6, margin_y - 20, max(bar_px + 12, 74), 26)
            painter.fillRect(bg_rect, QColor(5, 7, 11, 200))
            painter.setPen(QPen(QColor("#00f5d4"), 3.0))
            painter.drawLine(margin_x, margin_y, margin_x + bar_px, margin_y)
            painter.drawLine(margin_x, margin_y - 4, margin_x, margin_y + 2)
            painter.drawLine(margin_x + bar_px, margin_y - 4, margin_x + bar_px, margin_y + 2)
            painter.setPen(QColor("#e2e8f0"))
            painter.setFont(QFont("Consolas", 9, QFont.Bold))
            painter.drawText(margin_x, margin_y - 7, f"{bar_mm:g} mm")
            painter.end()

        self._lbl_slice_image.setPixmap(pix)
        self._lbl_slice_metrics.setText(
            f"Plane {res.axis} = {res.slice_coord_mm:.2f} mm | "
            f"Slice Domain: {res.width_mm:.2f} x {res.height_mm:.2f} mm | "
            f"Eff Pitch: {res.pixel_size_mm:.4f} mm | "
            f"Solid Area Fraction: {res.solid_area_fraction:.3f}"
        )

        prof = evaluate_grading_profile_1d(self.state, self._cad_mesh, n_samples=160)
        self._profile_chart.set_profile(
            prof,
            slice_coord_mm=res.slice_coord_mm,
            slice_axis=res.axis,
        )
        L_arr = prof["L_mm"]
        w_arr = prof["w_mm"]
        sf_arr = prof["sf"]
        n_marks = len(prof.get("markers_mm", []))
        self._lbl_profile_summary.setText(
            f"Mode: {self.state.grading_mode} | Axis: {prof['axis']} | Markers: {n_marks}\n"
            f"L: {float(np.min(L_arr)):.2f} .. {float(np.max(L_arr)):.2f} mm (mean {float(np.mean(L_arr)):.2f})\n"
            f"Wall w: {float(np.min(w_arr)):.3f} .. {float(np.max(w_arr)):.3f} mm | "
            f"SF: {float(np.min(sf_arr)):.3f} .. {float(np.max(sf_arr)):.3f}"
        )

    # ------------------------------------------------------------------
    # Background Workers: Calibration & Production Generation
    # ------------------------------------------------------------------
    def _on_calibrate_clicked(self) -> None:
        if self._cal_thread is not None and self._cal_thread.isRunning():
            return
        self._pull_state_from_widgets()
        self._btn_calibrate.setEnabled(False)
        self._lbl_cal_status.setText("Running exact LUT + Newton calibration...")

        self._cal_thread = QThread(self)
        worker = _CalibrationWorker(self.state)
        worker.moveToThread(self._cal_thread)
        self._cal_worker = worker
        self._cal_thread.started.connect(worker.run)
        worker.progress.connect(self._log)
        worker.finished.connect(self._on_calibration_finished)
        worker.failed.connect(self._on_calibration_failed)
        worker.finished.connect(self._cal_thread.quit)
        worker.failed.connect(self._cal_thread.quit)
        self._cal_thread.start()

    def _on_calibration_finished(self, out: CalibrationOutcome) -> None:
        self._btn_calibrate.setEnabled(True)
        self._updating_ui = True
        try:
            if out.updated_knots is not None:
                self.state.multi_knot_knots = out.updated_knots
                self._populate_knot_table()
            else:
                self.state.unit_cell_size_mm = out.calibrated_L_mm
                self.state.wall_thickness_mm = out.wall_thickness_mm
                self.state.solid_fraction = out.solid_fraction
                self.state.pore_size_mm = out.measured_pore_mm
                self._spin_L.setValue(out.calibrated_L_mm)
                self._spin_wall.setValue(out.wall_thickness_mm)
                self._spin_sf.setValue(out.solid_fraction)
                self._spin_pore.setValue(out.measured_pore_mm)
        finally:
            self._updating_ui = False

        badge = "Converged" if out.converged else "Best Fit"
        self._lbl_cal_status.setText(
            f"{badge} ({out.source}): L={out.calibrated_L_mm:.3f} mm, "
            f"MIS Pore={out.measured_pore_mm:.3f} mm "
            f"(err={out.residual_pore_mm:+.3f} mm, {out.iterations} iters), "
            f"SF={out.solid_fraction:.3f}, tau={out.calibrated_tau:.4f}"
        )
        self._lbl_cal_status.setStyleSheet("color: #34d399;" if out.converged else "color: #fbbf24;")
        self._log(
            f"Calibration complete ({out.source}): err={out.residual_pore_mm:.3f} mm, "
            f"iters={out.iterations}."
        )
        self._update_telemetry_label()
        self.refresh_visual_previews()

    def _on_calibration_failed(self, err: str) -> None:
        self._btn_calibrate.setEnabled(True)
        self._lbl_cal_status.setText(f"Calibration failed: {err}")
        self._lbl_cal_status.setStyleSheet("color: #f87171;")
        self._log(f"ERROR (Calibration): {err}")

    def _on_generate_clicked(self) -> None:
        if self._gen_thread is not None and self._gen_thread.isRunning():
            return
        self._pull_state_from_widgets()
        self._btn_generate.setEnabled(False)
        self._progress.setValue(0)
        self._log("=== Starting Production Implicit Generation ===")

        self._gen_thread = QThread(self)
        worker = _GenerationWorker(self.state, self._cad_mesh)
        worker.moveToThread(self._gen_thread)
        self._gen_worker = worker
        self._gen_thread.started.connect(worker.run)
        worker.progress.connect(self._on_gen_progress)
        worker.finished.connect(self._on_gen_finished)
        worker.failed.connect(self._on_gen_failed)
        worker.finished.connect(self._gen_thread.quit)
        worker.failed.connect(self._gen_thread.quit)
        self._gen_thread.start()

    def _on_gen_progress(self, pct: int, msg: str) -> None:
        self._progress.setValue(pct)
        self._log(f"[{pct:3d}%] {msg}")

    def _on_gen_finished(self, res: GenerationResult) -> None:
        self._btn_generate.setEnabled(True)
        self._last_result = res
        self._generated_mesh = res.mesh
        self._btn_save_copy.setEnabled(True)

        wt = res.health.get("is_watertight", False)
        faces = res.health.get("face_count", 0)
        nm_e = res.health.get("non_manifold_edges", 0)
        written_lines = "\n".join(
            f"  - {p.name[:40]}..." if len(p.name) > 43 else f"  - {p.name}"
            for p in res.output_paths
        ) or "  (memory only)"
        mis_line = (
            f"\nPost-MIS Pore: {res.mis_pore_mm:.3f} mm ({res.mis_confidence})"
            if res.mis_pore_mm is not None
            else ""
        )
        self._lbl_export_summary.setText(
            f"{'WATERTIGHT MANIFOLD' if wt else 'CHECK MANIFOLD'} | {faces:,} faces | NM edges={nm_e}\n"
            f"Time: {res.elapsed_seconds:.1f}s | Files:\n{written_lines}{mis_line}"
        )
        self._lbl_export_summary.setStyleSheet(
            "font-family: Consolas, monospace; color: #34d399;"
            if wt
            else "font-family: Consolas, monospace; color: #fbbf24;"
        )

        # Optional 3D mesh display: ONLY render if user explicitly checked the toggle
        if self._scene is not None:
            if self.state.show_generated_3d_mesh:
                self._scene.set_generated_mesh(res.mesh, visible=True)
                self._render_vtk()
            else:
                self._scene.set_generated_mesh(None, visible=False)
                self._log("3D mesh inspection toggle is OFF — exported to disk cleanly.")

    def _on_gen_failed(self, err: str) -> None:
        self._btn_generate.setEnabled(True)
        self._progress.setValue(0)
        self._log(f"ERROR (Generation): {err}")
        QMessageBox.critical(self, "Generation Failed", err)

    # ------------------------------------------------------------------
    # Recipe & Output File Actions
    # ------------------------------------------------------------------
    def _on_save_recipe(self) -> None:
        self._pull_state_from_widgets()
        DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Implicit Lattice Recipe",
            str(DEFAULT_OUTPUT_ROOT / "implicit_recipe.json"),
            "JSON Recipe (*.json)",
        )
        if path:
            self.state.save_recipe(path)
            self._log(f"Saved recipe: {path}")

    def _on_load_recipe(self) -> None:
        DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Implicit Lattice Recipe",
            str(DEFAULT_OUTPUT_ROOT),
            "JSON Recipe (*.json)",
        )
        if path:
            self.state = ImplicitStudioState.load_recipe(path)
            self._sync_widgets_from_state()
            self._rebuild_cad_and_refresh()
            self._log(f"Loaded recipe: {path}")

    def _on_batch_recipes_clicked(self) -> None:
        if self._batch_thread is not None and self._batch_thread.isRunning():
            return
        DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Recipes for Sequential Batch Run",
            str(DEFAULT_OUTPUT_ROOT),
            "JSON Recipes (*.json)",
        )
        if not paths:
            return
        self._btn_generate.setEnabled(False)
        self._progress.setValue(0)
        self._log(f"Starting batch queue ({len(paths)} recipe(s))...")

        self._batch_thread = QThread(self)
        worker = _BatchWorker(paths)
        worker.moveToThread(self._batch_thread)
        self._batch_worker = worker
        self._batch_thread.started.connect(worker.run)
        worker.progress.connect(self._on_gen_progress)
        worker.finished.connect(self._on_batch_finished)
        worker.failed.connect(self._on_gen_failed)
        worker.finished.connect(self._batch_thread.quit)
        worker.failed.connect(self._batch_thread.quit)
        self._batch_thread.start()

    def _on_batch_finished(self, results: list[tuple[Path, GenerationResult]]) -> None:
        self._btn_generate.setEnabled(True)
        self._progress.setValue(100)
        if results:
            self._last_result = results[-1][1]
            self._generated_mesh = results[-1][1].mesh
            self._btn_save_copy.setEnabled(True)
            lines = "\n".join(
                f"  - {rpath.name} -> {', '.join(p.name for p in res.output_paths)}"
                for rpath, res in results
            )
            self._lbl_export_summary.setText(f"Batch Complete ({len(results)} jobs):\n{lines}")
            self._lbl_export_summary.setStyleSheet("font-family: Consolas, monospace; color: #34d399;")
            self._log(f"Batch queue finished ({len(results)} jobs).")

    def _on_snapshot_clicked(self) -> None:
        snap_dir = DEFAULT_OUTPUT_ROOT.parent / "implicit_studio" / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arch = self.state.lattice_type.lower().replace(" ", "_")
        out = self.capture_window_png(snap_dir / f"implicit_studio_{arch}_{stamp}.png")
        self._log(f"Saved snapshot: {out}")

    def _on_open_output_folder(self) -> None:
        target = (
            self._last_result.output_dir
            if self._last_result is not None
            else DEFAULT_OUTPUT_ROOT
        )
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))

    def _on_save_copy_as(self) -> None:
        if self._generated_mesh is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Copy of Generated Mesh",
            str(DEFAULT_OUTPUT_ROOT / "lattice_copy.stl"),
            "STL Mesh (*.stl);;3MF Mesh (*.3mf)",
        )
        if path:
            self._generated_mesh.export(path)
            self._log(f"Saved copy: {path}")

    # ------------------------------------------------------------------
    # Remote Review Capture & Qt Lifecycle
    # ------------------------------------------------------------------
    def capture_window_png(self, output_path: str | Path) -> Path:
        """Capture full Qt studio window compositing live VTK OpenGL framebuffer."""
        import io
        from PIL import Image

        QApplication.processEvents()
        if self._vtk_widget is not None and self._tabs.currentIndex() == 0:
            self._vtk_widget.GetRenderWindow().Render()
            QApplication.processEvents()

        pixmap = self.grab()
        qbuf = QBuffer()
        qbuf.open(QIODevice.WriteOnly)
        pixmap.save(qbuf, "PNG")
        win_img = Image.open(io.BytesIO(bytes(qbuf.data()))).convert("RGBA")

        if self._vtk_widget is not None and self._tabs.currentIndex() == 0:
            rw = self._vtk_widget.GetRenderWindow()
            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(rw)
            w2i.SetInputBufferTypeToRGB()
            w2i.ReadFrontBufferOff()
            w2i.Update()

            png_writer = vtk.vtkPNGWriter()
            png_writer.WriteToMemoryOn()
            png_writer.SetInputConnection(w2i.GetOutputPort())
            png_writer.Write()
            vtk_bytes = bytes(memoryview(png_writer.GetResult()))
            vtk_img = Image.open(io.BytesIO(vtk_bytes)).convert("RGBA")

            top_left = self._vtk_widget.mapTo(self, QPoint(0, 0))
            dpr = float(self.devicePixelRatioF())
            target_w = int(round(self._vtk_widget.width() * dpr))
            target_h = int(round(self._vtk_widget.height() * dpr))
            if target_w > 0 and target_h > 0:
                if vtk_img.size != (target_w, target_h):
                    vtk_img = vtk_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                win_img.paste(
                    vtk_img,
                    (int(round(top_left.x() * dpr)), int(round(top_left.y() * dpr))),
                )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        win_img.convert("RGB").save(out)
        return out

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._vtk_widget is not None and not self._vtk_initialized:
            self._vtk_initialized = True
            self._vtk_widget.Initialize()
            if self._scene is not None:
                self._scene.attach_interactor(self._vtk_widget)
            self._render_vtk()

    def closeEvent(self, event) -> None:
        self._preview_timer.stop()
        self._cutaway_cap_timer.stop()
        self._slice_timer.stop()
        if self._vtk_widget is not None:
            try:
                self._vtk_widget.Finalize()
            except Exception:
                pass
        super().closeEvent(event)


def launch_implicit_studio(
    argv: list[str] | None = None,
    *,
    enable_vtk: bool = True,
    show: bool = True,
) -> int:
    """Launch the standalone Graphite Implicit Studio desktop application."""
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(argv if argv is not None else sys.argv)
    win = ImplicitStudioWindow(enable_vtk=enable_vtk)
    if show:
        win.show()
    if owns_app:
        return app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(launch_implicit_studio())
