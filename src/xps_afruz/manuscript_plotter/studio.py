"""XPS-master Manuscript Plotter - editable publication-quality XPS figures."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys
from dataclasses import fields
from pathlib import Path
import tkinter as tk
from tkinter import colorchooser, filedialog, font as tkfont, messagebox, ttk

from .core import (
    AnnotationSpec,
    MIN_EXPORT_DPI,
    PlotSettings,
    TraceStyle,
    apply_trace_template,
    auto_annotations,
    default_trace_styles,
    list_xlsx_sheets,
    load_xps_sheet,
    make_figure,
    planned_export_paths,
    project_from_json,
    project_to_json,
    safe_filename_part,
    save_figure,
    template_from_json,
    template_to_json,
)


APP_NAME = "XPS-master Manuscript Plotter"
APP_VERSION = "2.0"

EXPORT_FILETYPES = [
    ("PNG image", "*.png"),
    ("PDF vector", "*.pdf"),
    ("SVG vector", "*.svg"),
    ("TIFF image", "*.tif *.tiff"),
]


def _float_or_none(value: str):
    value = value.strip()
    return None if value == "" else float(value)


class XPSPlotStudio(tk.Tk):
    def __init__(self, initial_file: str | None = None):
        super().__init__()
        self.title(f"{APP_NAME} {APP_VERSION}")
        self.geometry("1450x900")
        self.minsize(1120, 720)
        self.option_add("*tearOff", False)
        self.data = None
        self.traces: list[TraceStyle] = []
        self.annotations: list[AnnotationSpec] = []
        self.settings = PlotSettings()
        self.source_path = ""
        self.figure = None
        self.ax = None
        self.canvas = None
        self.toolbar = None
        self.artists = {}
        self.dragging_annotation = None
        # Each worksheet keeps the exact state last shown in the GUI.  Batch
        # export uses these states and the same live Tk canvas as normal export.
        self.sheet_states: dict[str, dict] = {}
        self._build_style()
        self._build_menu()
        self._build_layout()
        self._bind_shortcuts()
        if initial_file:
            self.after(150, lambda: self.open_workbook(initial_file))

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 6))
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=25)

    def _build_menu(self):
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu)
        file_menu.add_command(label="Open Excel…", accelerator="Ctrl+O", command=self.open_workbook)
        file_menu.add_command(label="Open project…", command=self.open_project)
        file_menu.add_separator()
        file_menu.add_command(label="Save project…", accelerator="Ctrl+S", command=self.save_project)
        file_menu.add_command(label="Export figure…", accelerator="Ctrl+E", command=self.export_figure)
        file_menu.add_command(label="EXPORT ALL WORKBOOK PLOTS…", accelerator="Ctrl+Shift+E", command=self.export_all_figures)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menu.add_cascade(label="File", menu=file_menu)
        export_menu = tk.Menu(menu)
        export_menu.add_command(label="Export current plot…", accelerator="Ctrl+E", command=self.export_figure)
        export_menu.add_command(label="EXPORT ALL WORKBOOK PLOTS…", accelerator="Ctrl+Shift+E", command=self.export_all_figures)
        menu.add_cascade(label="Export", menu=export_menu)
        template_menu = tk.Menu(menu)
        template_menu.add_command(label="Save current style as template…", command=self.save_template)
        template_menu.add_command(label="Apply template to ALL plots…", command=self.open_template)
        menu.add_cascade(label="Templates", menu=template_menu)
        edit_menu = tk.Menu(menu)
        edit_menu.add_command(label="Apply reference-style preset", command=self.apply_preset)
        edit_menu.add_command(label="Auto-label component peaks", command=self.auto_label_peaks)
        edit_menu.add_command(label="Clear annotations", command=self.clear_annotations)
        menu.add_cascade(label="Edit", menu=edit_menu)
        help_menu = tk.Menu(menu)
        help_menu.add_command(label="Quick help", command=self.show_help)
        help_menu.add_command(
            label="About",
            command=lambda: messagebox.showinfo(
                "About",
                f"{APP_NAME} {APP_VERSION}\nAvantage and XPS-master workbooks with fitted components and residuals.",
            ),
        )
        menu.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menu)

    def _build_layout(self):
        top = ttk.Frame(self, padding=(8, 7))
        top.pack(fill="x")
        ttk.Button(top, text="Open Excel", command=self.open_workbook).pack(side="left", padx=(0, 5))
        ttk.Button(top, text="Update Plot", style="Accent.TButton", command=self.update_plot).pack(side="left", padx=5)
        ttk.Button(top, text="Reference Preset", command=self.apply_preset).pack(side="left", padx=5)
        ttk.Button(top, text="Auto Peak Labels", command=self.auto_label_peaks).pack(side="left", padx=5)
        ttk.Button(top, text="Export current", command=self.export_figure).pack(side="left", padx=5)
        ttk.Button(top, text="EXPORT ALL SHEETS", style="Accent.TButton", command=self.export_all_figures).pack(side="left", padx=5)
        self.status_var = tk.StringVar(value="Open an Excel workbook to begin.")
        ttk.Label(top, textvariable=self.status_var, anchor="e").pack(side="right", fill="x", expand=True, padx=8)

        pane = ttk.Panedwindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True)
        left = ttk.Frame(pane, width=430)
        right = ttk.Frame(pane)
        pane.add(left, weight=0)
        pane.add(right, weight=1)
        self.controls = ttk.Notebook(left)
        self.controls.pack(fill="both", expand=True, padx=(8, 4), pady=(0, 8))
        self.plot_host = right
        self._build_data_tab()
        self._build_plot_tab()
        self._build_annotation_tab()
        self._show_empty_canvas()

    def _scroll_tab(self, title):
        outer = ttk.Frame(self.controls)
        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=8)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.controls.add(outer, text=title)
        return inner

    def _build_data_tab(self):
        tab = self._scroll_tab("Data & Traces")
        ttk.Label(tab, text="Workbook and scan", style="Section.TLabel").pack(anchor="w")
        self.file_var = tk.StringVar(value="No workbook loaded")
        ttk.Label(tab, textvariable=self.file_var, wraplength=380, foreground="#555555").pack(anchor="w", fill="x", pady=(3, 6))
        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Label(row, text="Sheet:").pack(side="left")
        self.sheet_var = tk.StringVar()
        self.sheet_combo = ttk.Combobox(row, textvariable=self.sheet_var, state="readonly", width=25)
        self.sheet_combo.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Load", command=self.load_selected_sheet).pack(side="right")
        self.mapping_var = tk.StringVar(value="")
        ttk.Label(tab, textvariable=self.mapping_var, wraplength=385, justify="left", foreground="#666666").pack(anchor="w", pady=(6, 10))

        export_box = ttk.LabelFrame(tab, text="Export", padding=8)
        export_box.pack(fill="x", pady=(0, 10))
        ttk.Button(export_box, text="Export current plot…", command=self.export_figure).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(export_box, text="EXPORT ALL SHEETS…", style="Accent.TButton", command=self.export_all_figures).pack(side="left", fill="x", expand=True, padx=(4, 0))

        proc = ttk.LabelFrame(tab, text="Data processing", padding=8)
        proc.pack(fill="x", pady=(0, 10))
        self.subtract_var = tk.BooleanVar(value=self.settings.subtract_background)
        ttk.Checkbutton(proc, text="Subtract fitted background", variable=self.subtract_var, command=self.update_plot).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(proc, text="Normalize:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.normalize_var = tk.StringVar(value=self.settings.normalization)
        combo = ttk.Combobox(proc, textvariable=self.normalize_var, values=["None", "Maximum = 1", "Maximum = 100"], state="readonly", width=18)
        combo.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(5, 0))
        combo.bind("<<ComboboxSelected>>", lambda _e: self.update_plot())
        proc.columnconfigure(1, weight=1)

        ttk.Label(tab, text="Traces", style="Section.TLabel").pack(anchor="w")
        columns = ("show", "kind", "name", "color")
        self.trace_tree = ttk.Treeview(tab, columns=columns, show="headings", height=10, selectmode="browse")
        for key, title, width in [("show", "On", 38), ("kind", "Type", 74), ("name", "Name", 170), ("color", "Color", 72)]:
            self.trace_tree.heading(key, text=title)
            self.trace_tree.column(key, width=width, minwidth=35, stretch=key == "name")
        self.trace_tree.pack(fill="x", pady=(3, 5))
        self.trace_tree.bind("<<TreeviewSelect>>", self.on_trace_select)
        self.trace_tree.bind("<Double-1>", self.on_trace_double_click)

        editor = ttk.LabelFrame(tab, text="Selected trace style", padding=8)
        editor.pack(fill="x")
        self.trace_name_var = tk.StringVar()
        self.trace_visible_var = tk.BooleanVar()
        self.trace_fill_var = tk.BooleanVar()
        self.trace_width_var = tk.DoubleVar(value=1)
        self.trace_alpha_var = tk.DoubleVar(value=1)
        self.trace_fill_alpha_var = tk.DoubleVar(value=.55)
        self.trace_gradient_var = tk.DoubleVar(value=.55)
        self.trace_style_var = tk.StringVar(value="-")
        ttk.Label(editor, text="Display name").grid(row=0, column=0, sticky="w")
        ttk.Entry(editor, textvariable=self.trace_name_var).grid(row=0, column=1, columnspan=3, sticky="ew", padx=(5, 0))
        ttk.Checkbutton(editor, text="Visible", variable=self.trace_visible_var).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Checkbutton(editor, text="Fill component", variable=self.trace_fill_var).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Button(editor, text="Choose color…", command=self.choose_trace_color).grid(row=1, column=2, columnspan=2, sticky="e")
        self._grid_spin(editor, 2, "Line width", self.trace_width_var, 0.05, 12, .1)
        self._grid_spin(editor, 3, "Line opacity", self.trace_alpha_var, 0, 1, .05)
        self._grid_spin(editor, 4, "Fill opacity", self.trace_fill_alpha_var, 0, 1, .05)
        self._grid_spin(editor, 5, "Gradient ratio", self.trace_gradient_var, 0, 1, .05)
        ttk.Label(editor, text="Line style").grid(row=6, column=0, sticky="w", pady=3)
        ttk.Combobox(editor, textvariable=self.trace_style_var, values=["-", "--", "-.", ":", "None"], state="readonly", width=9).grid(row=6, column=1, sticky="w")
        ttk.Button(editor, text="Apply trace style", command=self.apply_trace_editor).grid(row=7, column=0, columnspan=4, sticky="ew", pady=(7, 0))
        editor.columnconfigure(1, weight=1)
        editor.columnconfigure(3, weight=1)

    def _grid_spin(self, parent, row, label, variable, low, high, increment):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Spinbox(parent, textvariable=variable, from_=low, to=high, increment=increment, width=9).grid(row=row, column=1, sticky="w")

    def _build_plot_tab(self):
        tab = self._scroll_tab("Plot & Axes")
        self.plot_vars = {}
        text_box = ttk.LabelFrame(tab, text="Text", padding=8)
        text_box.pack(fill="x", pady=(0, 9))
        for row, (key, label, value) in enumerate([
            ("title", "Panel title", self.settings.title),
            ("x_label", "X-axis label", self.settings.x_label),
            ("y_label", "Y-axis label", self.settings.y_label),
            ("font_family", "Font family", self.settings.font_family),
        ]):
            var = tk.StringVar(value=value)
            self.plot_vars[key] = var
            ttk.Label(text_box, text=label).grid(row=row, column=0, sticky="w", pady=3)
            if key == "font_family":
                fonts = sorted(set(tkfont.families()))
                ttk.Combobox(text_box, textvariable=var, values=fonts, width=25).grid(row=row, column=1, sticky="ew", padx=(6, 0))
            else:
                ttk.Entry(text_box, textvariable=var).grid(row=row, column=1, sticky="ew", padx=(6, 0))
        text_box.columnconfigure(1, weight=1)

        numeric = ttk.LabelFrame(tab, text="Typography and lines", padding=8)
        numeric.pack(fill="x", pady=(0, 9))
        items = [
            ("title_size", "Title size", self.settings.title_size, 6, 60, 1),
            ("label_size", "Axis label size", self.settings.label_size, 6, 60, 1),
            ("tick_size", "Tick number size", self.settings.tick_size, 6, 50, 1),
            ("axis_width", "Axes thickness", self.settings.axis_width, .2, 10, .1),
            ("tick_width", "Tick thickness", self.settings.tick_width, .2, 10, .1),
            ("tick_length", "Major tick length", self.settings.tick_length, 0, 30, .5),
            ("minor_tick_length", "Minor tick length", self.settings.minor_tick_length, 0, 20, .5),
        ]
        for row, (key, label, value, low, high, inc) in enumerate(items):
            var = tk.DoubleVar(value=value)
            self.plot_vars[key] = var
            ttk.Label(numeric, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Spinbox(numeric, textvariable=var, from_=low, to=high, increment=inc, width=10).grid(row=row, column=1, sticky="e")
        numeric.columnconfigure(0, weight=1)

        canvas_box = ttk.LabelFrame(tab, text="Canvas and axes", padding=8)
        canvas_box.pack(fill="x", pady=(0, 9))
        for row, (key, label, value) in enumerate([
            ("figure_width", "Figure width (inch)", self.settings.figure_width),
            ("figure_height", "Figure height (inch)", self.settings.figure_height),
            ("dpi", "Export DPI", self.settings.dpi),
        ]):
            var = tk.IntVar(value=value) if key == "dpi" else tk.DoubleVar(value=value)
            self.plot_vars[key] = var
            ttk.Label(canvas_box, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Spinbox(
                canvas_box,
                textvariable=var,
                from_=MIN_EXPORT_DPI if key == "dpi" else 2,
                to=2400 if key == "dpi" else 30,
                increment=100 if key == "dpi" else .1,
                width=10,
            ).grid(row=row, column=1, sticky="e")
        bools = [
            ("reverse_x", "Reverse binding-energy axis", self.settings.reverse_x),
            ("square_axes", "Perfect square plotting area", self.settings.square_axes),
            ("show_y_tick_labels", "Show Y-axis numbers", self.settings.show_y_tick_labels),
            ("show_minor_ticks", "Show minor ticks", self.settings.show_minor_ticks),
            ("show_legend", "Show legend", self.settings.show_legend),
            ("show_grid", "Show grid", self.settings.show_grid),
            ("show_residual_panel", "Separate residual panel", self.settings.show_residual_panel),
            ("transparent_export", "Transparent exported background", self.settings.transparent_export),
        ]
        base_row = 3
        for offset, (key, label, value) in enumerate(bools):
            var = tk.BooleanVar(value=value)
            self.plot_vars[key] = var
            ttk.Checkbutton(canvas_box, text=label, variable=var).grid(row=base_row + offset, column=0, columnspan=2, sticky="w", pady=1)
        canvas_box.columnconfigure(0, weight=1)

        limits = ttk.LabelFrame(tab, text="Manual limits (blank = automatic)", padding=8)
        limits.pack(fill="x", pady=(0, 9))
        for row, (key, label, value) in enumerate([
            ("x_min", "X minimum", self.settings.x_min), ("x_max", "X maximum", self.settings.x_max),
            ("y_min", "Y minimum", self.settings.y_min), ("y_max", "Y maximum", self.settings.y_max),
        ]):
            var = tk.StringVar(value="" if value is None else str(value))
            self.plot_vars[key] = var
            ttk.Label(limits, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(limits, textvariable=var, width=13).grid(row=row, column=1, sticky="e")
        limits.columnconfigure(0, weight=1)
        ttk.Button(tab, text="Apply all plot settings", style="Accent.TButton", command=self.update_plot).pack(fill="x")
        template_bar = ttk.Frame(tab)
        template_bar.pack(fill="x", pady=(7, 0))
        ttk.Button(template_bar, text="Save Template…", command=self.save_template).pack(side="left", fill="x", expand=True, padx=(0, 3))
        ttk.Button(template_bar, text="Apply to ALL Plots…", command=self.open_template).pack(side="left", fill="x", expand=True, padx=(3, 0))

    def _build_annotation_tab(self):
        tab = self._scroll_tab("Labels")
        ttk.Label(tab, text="Annotations can use Matplotlib math text, e.g. $Fe^{3+}$.", wraplength=385).pack(anchor="w", pady=(0, 8))
        self.annotation_tree = ttk.Treeview(tab, columns=("text", "x", "y"), show="headings", height=10)
        for key, label, width in [("text", "Text", 170), ("x", "X", 80), ("y", "Y", 90)]:
            self.annotation_tree.heading(key, text=label)
            self.annotation_tree.column(key, width=width, stretch=key == "text")
        self.annotation_tree.pack(fill="x")
        self.annotation_tree.bind("<<TreeviewSelect>>", self.on_annotation_select)
        bar = ttk.Frame(tab)
        bar.pack(fill="x", pady=5)
        ttk.Button(bar, text="Add", command=self.add_annotation).pack(side="left")
        ttk.Button(bar, text="Auto peaks", command=self.auto_label_peaks).pack(side="left", padx=4)
        ttk.Button(bar, text="Delete", command=self.delete_annotation).pack(side="left")
        ttk.Button(bar, text="Clear", command=self.clear_annotations).pack(side="right")

        editor = ttk.LabelFrame(tab, text="Selected label", padding=8)
        editor.pack(fill="x", pady=(4, 0))
        self.ann_vars = {
            "text": tk.StringVar(), "x": tk.StringVar(), "y": tk.StringVar(),
            "rotation": tk.DoubleVar(value=90), "font_size": tk.DoubleVar(value=12),
            "color": tk.StringVar(value="#111111"), "bold": tk.BooleanVar(value=False),
        }
        for row, (key, label) in enumerate([("text", "Text"), ("x", "X position"), ("y", "Y position"), ("rotation", "Rotation (degrees)"), ("font_size", "Font size")]):
            ttk.Label(editor, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(editor, textvariable=self.ann_vars[key]).grid(row=row, column=1, sticky="ew", padx=(6, 0))
        ttk.Checkbutton(editor, text="Bold", variable=self.ann_vars["bold"]).grid(row=5, column=0, sticky="w")
        ttk.Button(editor, text="Label color…", command=self.choose_annotation_color).grid(row=5, column=1, sticky="e")
        ttk.Button(editor, text="Apply label", command=self.apply_annotation_editor).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        editor.columnconfigure(1, weight=1)
        ttk.Label(tab, text="Tip: drag a label directly on the plot to reposition it.", foreground="#555555", wraplength=385).pack(anchor="w", pady=8)

    def _show_empty_canvas(self):
        label = ttk.Label(
            self.plot_host,
            text="Open an Avantage or XPS-master Excel workbook\nto create a manuscript-ready figure.",
            font=("Segoe UI", 16),
            anchor="center",
            justify="center",
        )
        label.pack(fill="both", expand=True)

    def _bind_shortcuts(self):
        self.bind_all("<Control-o>", lambda _e: self.open_workbook())
        self.bind_all("<Control-s>", lambda _e: self.save_project())
        self.bind_all("<Control-e>", lambda _e: self.export_figure())
        self.bind_all("<Control-Shift-E>", lambda _e: self.export_all_figures())

    @staticmethod
    def _default_sheet_title(sheet_name: str) -> str:
        return sheet_name.replace(" Scan", "")

    def _capture_sheet_state(self) -> dict:
        """Capture settings plus the live toolbar view for the current sheet."""
        state = {
            "settings": deepcopy(self.settings),
            "traces": deepcopy(self.traces),
            "annotations": deepcopy(self.annotations),
            "xlim": None,
            "ylim": None,
        }
        if self.ax is not None:
            state["xlim"] = tuple(float(value) for value in self.ax.get_xlim())
            state["ylim"] = tuple(float(value) for value in self.ax.get_ylim())
        return state

    def _remember_current_sheet(self):
        if self.data is not None:
            self.sheet_states[self.data.sheet_name] = self._capture_sheet_state()

    def _show_sheet_state(self, data, state: dict):
        """Render one worksheet through the real GUI canvas."""
        self.data = data
        self.sheet_var.set(data.sheet_name)
        self.settings = deepcopy(state["settings"])
        self.traces = apply_trace_template(default_trace_styles(data), deepcopy(state["traces"]))
        self.annotations = deepcopy(state["annotations"])
        self._sync_settings_controls()
        self.refresh_trace_tree()
        self.refresh_annotation_tree()
        self.update_plot()
        if state.get("xlim") is not None:
            self.ax.set_xlim(state["xlim"])
        if state.get("ylim") is not None:
            self.ax.set_ylim(state["ylim"])
        self.canvas.draw()
        self.update_idletasks()
        self._remember_current_sheet()

    def open_workbook(self, path=None):
        if not path:
            path = filedialog.askopenfilename(title="Open XPS workbook", filetypes=[("Excel workbooks", "*.xlsx *.xlsm"), ("All files", "*.*")])
        if not path:
            return
        try:
            sheets = list_xlsx_sheets(path)
        except Exception as exc:
            messagebox.showerror("Cannot open workbook", str(exc))
            return
        self.source_path = str(Path(path))
        self.sheet_states.clear()
        self.data = None
        self.file_var.set(self.source_path)
        self.sheet_combo["values"] = sheets
        preferred = next((s for s in sheets if "scan" in s.lower()), sheets[0] if sheets else "")
        self.sheet_var.set(preferred)
        self.load_selected_sheet()

    def load_selected_sheet(self):
        if not self.source_path or not self.sheet_var.get():
            return
        target_sheet = self.sheet_var.get()
        previous_sheet = self.data.sheet_name if self.data is not None else ""
        previous_settings = deepcopy(self.settings)
        previous_traces = deepcopy(self.traces)
        self._remember_current_sheet()
        try:
            self.status_var.set(f"Loading {target_sheet}…")
            self.update_idletasks()
            data = load_xps_sheet(self.source_path, target_sheet)
            saved_state = self.sheet_states.get(target_sheet)
            if saved_state is not None:
                self._show_sheet_state(data, saved_state)
            else:
                settings = previous_settings
                if not previous_sheet:
                    settings.title = self._default_sheet_title(target_sheet)
                elif not settings.title or settings.title == self._default_sheet_title(previous_sheet):
                    settings.title = self._default_sheet_title(target_sheet)
                traces = apply_trace_template(default_trace_styles(data), previous_traces) if previous_traces else default_trace_styles(data)
                self._show_sheet_state(data, {
                    "settings": settings,
                    "traces": traces,
                    "annotations": [],
                    "xlim": None,
                    "ylim": None,
                })
            s = self.data.column_summary
            self.mapping_var.set(
                f"{s['format']}  |  {s['points']} points\n"
                f"Energy {s['energy']}  |  Measured {s['measured']}\n"
                f"Envelope {s['envelope']}  |  Background {s['background']}  |  Residual {s['residual']}\n"
                f"Component columns: {s['components']}"
            )
            self.refresh_trace_tree()
            self.refresh_annotation_tree()
            self.update_plot()
            self.status_var.set(f"Loaded {self.data.sheet_name}: {s['points']} points, {len(self.data.components)} component(s)")
        except Exception as exc:
            messagebox.showerror("Cannot load XPS sheet", str(exc))
            self.status_var.set("Load failed.")

    def collect_settings(self):
        try:
            for key in ["title", "x_label", "y_label", "font_family"]:
                setattr(self.settings, key, self.plot_vars[key].get())
            for key in ["title_size", "label_size", "tick_size", "axis_width", "tick_width", "tick_length", "minor_tick_length", "figure_width", "figure_height"]:
                setattr(self.settings, key, float(self.plot_vars[key].get()))
            self.settings.dpi = max(MIN_EXPORT_DPI, int(self.plot_vars["dpi"].get()))
            self.plot_vars["dpi"].set(self.settings.dpi)
            for key in [
                "reverse_x",
                "square_axes",
                "show_y_tick_labels",
                "show_minor_ticks",
                "show_legend",
                "show_grid",
                "show_residual_panel",
                "transparent_export",
            ]:
                setattr(self.settings, key, bool(self.plot_vars[key].get()))
            for key in ["x_min", "x_max", "y_min", "y_max"]:
                setattr(self.settings, key, _float_or_none(self.plot_vars[key].get()))
            self.settings.subtract_background = bool(self.subtract_var.get())
            self.settings.normalization = self.normalize_var.get()
            return True
        except (TypeError, ValueError) as exc:
            messagebox.showerror("Invalid setting", f"Please check the numeric plot settings.\n\n{exc}")
            return False

    def update_plot(self):
        if self.data is None or not self.collect_settings():
            return
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar:
            self.toolbar.destroy()
        for child in self.plot_host.winfo_children():
            child.destroy()
        self.figure, self.ax, self.artists = make_figure(self.data, self.settings, self.traces, self.annotations)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_host)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_host, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x")
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self._remember_current_sheet()

    def _on_pick(self, event):
        index = getattr(event.artist, "_xps_annotation_index", None)
        if index is not None:
            self.dragging_annotation = int(index)

    def _on_motion(self, event):
        if self.dragging_annotation is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        note = self.annotations[self.dragging_annotation]
        note.x, note.y = float(event.xdata), float(event.ydata)
        artist = self.artists.get("annotations", [])[self.dragging_annotation]
        artist.set_position((note.x, note.y))
        self.canvas.draw_idle()

    def _on_release(self, _event):
        if self.dragging_annotation is not None:
            self.dragging_annotation = None
            self.refresh_annotation_tree()

    def refresh_trace_tree(self):
        self.trace_tree.delete(*self.trace_tree.get_children())
        for index, item in enumerate(self.traces):
            self.trace_tree.insert("", "end", iid=str(index), values=("✓" if item.visible else "", item.kind, item.name, item.color))
        if self.traces:
            self.trace_tree.selection_set("0")
            self.on_trace_select()

    def selected_trace(self):
        selected = self.trace_tree.selection()
        return self.traces[int(selected[0])] if selected else None

    def on_trace_select(self, _event=None):
        item = self.selected_trace()
        if not item:
            return
        self.trace_name_var.set(item.name)
        self.trace_visible_var.set(item.visible)
        self.trace_fill_var.set(item.fill)
        self.trace_width_var.set(item.line_width)
        self.trace_alpha_var.set(item.alpha)
        self.trace_fill_alpha_var.set(item.fill_alpha)
        self.trace_gradient_var.set(item.gradient_ratio)
        self.trace_style_var.set(item.line_style)

    def on_trace_double_click(self, event):
        row = self.trace_tree.identify_row(event.y)
        col = self.trace_tree.identify_column(event.x)
        if row and col == "#1":
            self.traces[int(row)].visible = not self.traces[int(row)].visible
            self.refresh_trace_tree()
            self.update_plot()

    def choose_trace_color(self):
        item = self.selected_trace()
        if not item:
            return
        chosen = colorchooser.askcolor(item.color, title="Choose trace color")[1]
        if chosen:
            item.color = chosen
            self.refresh_trace_tree()
            self.update_plot()

    def apply_trace_editor(self):
        item = self.selected_trace()
        if not item:
            return
        try:
            item.name = self.trace_name_var.get()
            item.visible = bool(self.trace_visible_var.get())
            item.fill = bool(self.trace_fill_var.get())
            item.line_width = float(self.trace_width_var.get())
            item.alpha = float(self.trace_alpha_var.get())
            item.fill_alpha = float(self.trace_fill_alpha_var.get())
            item.gradient_ratio = float(self.trace_gradient_var.get())
            item.line_style = self.trace_style_var.get()
        except ValueError as exc:
            messagebox.showerror("Invalid trace style", str(exc))
            return
        self.refresh_trace_tree()
        self.update_plot()

    def refresh_annotation_tree(self):
        self.annotation_tree.delete(*self.annotation_tree.get_children())
        for index, note in enumerate(self.annotations):
            self.annotation_tree.insert("", "end", iid=str(index), values=(note.text, f"{note.x:.4g}", f"{note.y:.4g}"))

    def add_annotation(self):
        if self.ax is None:
            return
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.annotations.append(AnnotationSpec("Label", sum(xlim) / 2, sum(ylim) / 2, 90, 12))
        self.refresh_annotation_tree()
        index = len(self.annotations) - 1
        self.annotation_tree.selection_set(str(index))
        self.on_annotation_select()
        self.update_plot()

    def auto_label_peaks(self):
        if self.data is None:
            return
        self.collect_settings()
        self.annotations = auto_annotations(self.data, self.settings, self.traces)
        self.refresh_annotation_tree()
        self.update_plot()

    def clear_annotations(self):
        self.annotations = []
        self.refresh_annotation_tree()
        self.update_plot()

    def delete_annotation(self):
        selected = self.annotation_tree.selection()
        if not selected:
            return
        del self.annotations[int(selected[0])]
        self.refresh_annotation_tree()
        self.update_plot()

    def on_annotation_select(self, _event=None):
        selected = self.annotation_tree.selection()
        if not selected:
            return
        note = self.annotations[int(selected[0])]
        self.ann_vars["text"].set(note.text)
        self.ann_vars["x"].set(str(note.x))
        self.ann_vars["y"].set(str(note.y))
        self.ann_vars["rotation"].set(note.rotation)
        self.ann_vars["font_size"].set(note.font_size)
        self.ann_vars["color"].set(note.color)
        self.ann_vars["bold"].set(note.bold)

    def choose_annotation_color(self):
        chosen = colorchooser.askcolor(self.ann_vars["color"].get(), title="Choose label color")[1]
        if chosen:
            self.ann_vars["color"].set(chosen)

    def apply_annotation_editor(self):
        selected = self.annotation_tree.selection()
        if not selected:
            return
        try:
            note = self.annotations[int(selected[0])]
            note.text = self.ann_vars["text"].get()
            note.x = float(self.ann_vars["x"].get())
            note.y = float(self.ann_vars["y"].get())
            note.rotation = float(self.ann_vars["rotation"].get())
            note.font_size = float(self.ann_vars["font_size"].get())
            note.color = self.ann_vars["color"].get()
            note.bold = bool(self.ann_vars["bold"].get())
        except ValueError as exc:
            messagebox.showerror("Invalid annotation", str(exc))
            return
        self.refresh_annotation_tree()
        self.update_plot()

    def apply_preset(self):
        defaults = PlotSettings(title=self.data.sheet_name.replace(" Scan", "") if self.data else "")
        self.settings = defaults
        for field in fields(PlotSettings):
            key = field.name
            if key in self.plot_vars:
                value = getattr(defaults, key)
                self.plot_vars[key].set("" if value is None else value)
        self.subtract_var.set(True)
        self.normalize_var.set("None")
        if self.data:
            current_names = {item.key: item.name for item in self.traces}
            self.traces = default_trace_styles(self.data)
            for item in self.traces:
                if item.key in current_names:
                    item.name = current_names[item.key]
            self.refresh_trace_tree()
        self.update_plot()

    def export_figure(self):
        if self.data is None or self.figure is None or self.canvas is None:
            messagebox.showinfo("Nothing to export", "Open an XPS workbook first.")
            return
        try:
            export_dpi = max(MIN_EXPORT_DPI, int(self.plot_vars["dpi"].get()))
            self.plot_vars["dpi"].set(export_dpi)
            transparent = bool(self.plot_vars["transparent_export"].get())
        except (TypeError, ValueError) as exc:
            messagebox.showerror("Invalid export setting", str(exc))
            return
        path = filedialog.asksaveasfilename(title="Export manuscript figure", defaultextension=".png", filetypes=EXPORT_FILETYPES)
        if not path:
            return
        try:
            # Save the live canvas, rather than rebuilding a second figure.
            # This preserves exactly the previewed limits, margins, annotation
            # positions, aspect ratio, and toolbar zoom/pan state.
            self.canvas.draw()
            save_figure(self.figure, path, export_dpi, transparent)
            self.status_var.set(f"Exported {path}")
            messagebox.showinfo("Export complete", f"Saved the figure exactly as previewed:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def export_all_figures(self):
        """Export every worksheet from the same live canvas used by the GUI."""
        if self.data is None or self.figure is None or self.canvas is None:
            messagebox.showinfo("Nothing to export", "Open an XPS workbook first.")
            return
        try:
            # Match single-figure export: save the visible plot state, while
            # allowing DPI/transparency to be changed immediately before save.
            export_dpi = max(MIN_EXPORT_DPI, int(self.plot_vars["dpi"].get()))
            self.plot_vars["dpi"].set(export_dpi)
            transparent = bool(self.plot_vars["transparent_export"].get())
            sheets = list_xlsx_sheets(self.source_path)
        except (TypeError, ValueError, OSError) as exc:
            messagebox.showerror("Cannot prepare export", str(exc))
            return

        initial_name = f"{safe_filename_part(Path(self.source_path).stem)}_plots.png"
        base_path = filedialog.asksaveasfilename(
            title="Export all workbook plots — choose base name and format",
            initialfile=initial_name,
            defaultextension=".png",
            filetypes=EXPORT_FILETYPES,
        )
        if not base_path:
            return

        targets = planned_export_paths(base_path, sheets)
        existing = [path for path in targets.values() if path.exists()]
        if existing and not messagebox.askyesno(
            "Replace existing exports?",
            f"{len(existing)} export file(s) already exist and will be replaced. Continue?",
        ):
            return

        current_sheet = self.data.sheet_name
        current_state = self._capture_sheet_state()
        self.sheet_states[current_sheet] = deepcopy(current_state)
        current_default_title = self._default_sheet_title(current_sheet)
        adapt_title = not current_state["settings"].title or current_state["settings"].title == current_default_title
        use_auto_labels = bool(current_state["annotations"])
        exported: list[Path] = []
        skipped: list[tuple[str, str]] = []

        try:
            for index, sheet_name in enumerate(sheets, start=1):
                target = targets[sheet_name]
                self.status_var.set(f"Rendering in GUI and exporting {index}/{len(sheets)}: {sheet_name}…")
                self.update_idletasks()
                try:
                    data = load_xps_sheet(self.source_path, sheet_name)
                    state = self.sheet_states.get(sheet_name)
                    if state is None:
                        sheet_settings = deepcopy(current_state["settings"])
                        if adapt_title:
                            sheet_settings.title = self._default_sheet_title(sheet_name)
                        traces = apply_trace_template(default_trace_styles(data), current_state["traces"])
                        annotations = auto_annotations(data, sheet_settings, traces) if use_auto_labels else []
                        state = {
                            "settings": sheet_settings,
                            "traces": traces,
                            "annotations": annotations,
                            "xlim": None,
                            "ylim": None,
                        }

                    # Critical v1.4 fix: every sheet follows the identical path
                    # used when the user selects it in the interface.  There is
                    # no separate off-screen batch renderer.
                    self._show_sheet_state(data, state)
                    save_figure(self.figure, target, export_dpi, transparent)
                    exported.append(target)
                except Exception as exc:
                    skipped.append((sheet_name, str(exc)))
        finally:
            # Batch export must not leave the interface on another worksheet or
            # discard the active toolbar zoom/pan and dragged label positions.
            try:
                original_data = load_xps_sheet(self.source_path, current_sheet)
                self._show_sheet_state(original_data, current_state)
            except Exception as exc:
                skipped.append((current_sheet, f"Could not restore GUI state: {exc}"))

        if not exported:
            self.status_var.set("Batch export failed.")
            detail = "\n".join(f"• {sheet}: {error}" for sheet, error in skipped[:8])
            messagebox.showerror("Export failed", f"No workbook plots could be exported.\n\n{detail}")
            return

        self.status_var.set(f"Exported {len(exported)} GUI-matched plot(s) to {exported[0].parent}")
        summary = f"Saved {len(exported)} plot(s) as {Path(base_path).suffix.upper().lstrip('.')} files in:\n{exported[0].parent}"
        if skipped:
            summary += f"\n\nSkipped {len(skipped)} non-plot or unreadable sheet(s):\n" + "\n".join(f"• {sheet}" for sheet, _ in skipped[:10])
        summary += "\n\nEvery plot was rendered through the same live GUI canvas used for preview."
        messagebox.showinfo("Export all complete", summary)

    def save_template(self):
        if self.data is None or not self.collect_settings():
            messagebox.showinfo("Nothing to save", "Open an XPS workbook first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save XPS plot template",
            defaultextension=".xpstemplate",
            filetypes=[("XPS Plot Studio template", "*.xpstemplate"), ("JSON", "*.json")],
        )
        if not path:
            return
        try:
            Path(path).write_text(template_to_json(self.settings, self.traces), encoding="utf-8")
            self.status_var.set(f"Saved template {path}")
            messagebox.showinfo("Template saved", f"Saved reusable plot template:\n{path}")
        except Exception as exc:
            messagebox.showerror("Cannot save template", str(exc))

    def open_template(self):
        if self.data is None:
            messagebox.showinfo("No data loaded", "Open an XPS workbook before applying a template.")
            return
        path = filedialog.askopenfilename(
            title="Apply XPS plot template",
            filetypes=[("XPS Plot Studio template", "*.xpstemplate *.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            payload = template_from_json(Path(path).read_text(encoding="utf-8"))
            valid_settings = {field.name for field in fields(PlotSettings)}
            settings_data = {key: value for key, value in payload["settings"].items() if key in valid_settings}
            template_settings = PlotSettings(**settings_data)
            saved_traces = [TraceStyle(**item) for item in payload["traces"]]

            # A template is project-wide: apply it to every plottable workbook
            # sheet, not only to the sheet currently visible in the GUI.
            current_sheet = self.data.sheet_name
            self._remember_current_sheet()
            template_default_title = self._default_sheet_title(current_sheet)
            adapt_title = not template_settings.title or template_settings.title == template_default_title
            applied = 0
            skipped: list[tuple[str, str]] = []

            for sheet_name in list_xlsx_sheets(self.source_path):
                try:
                    data = load_xps_sheet(self.source_path, sheet_name)
                    old_state = self.sheet_states.get(sheet_name)
                    sheet_settings = deepcopy(template_settings)
                    if adapt_title:
                        sheet_settings.title = self._default_sheet_title(sheet_name)
                    self.sheet_states[sheet_name] = {
                        "settings": sheet_settings,
                        "traces": apply_trace_template(default_trace_styles(data), saved_traces),
                        # Templates contain style only. Keep sheet-specific labels
                        # and view limits already created by the user.
                        "annotations": deepcopy(old_state["annotations"]) if old_state else [],
                        "xlim": deepcopy(old_state.get("xlim")) if old_state else None,
                        "ylim": deepcopy(old_state.get("ylim")) if old_state else None,
                    }
                    applied += 1
                except Exception as exc:
                    skipped.append((sheet_name, str(exc)))

            if applied == 0:
                raise ValueError("The workbook contains no plottable XPS worksheets.")

            # Re-render the originally selected sheet with its newly templated
            # state, leaving the user on the same plot and view.
            current_data = load_xps_sheet(self.source_path, current_sheet)
            self._show_sheet_state(current_data, self.sheet_states[current_sheet])
            self.status_var.set(f"Applied template to {applied} plot(s): {path}")
            summary = f"Applied the template to {applied} plot(s) in the current project."
            if skipped:
                summary += f"\n\nSkipped {len(skipped)} non-plot or unreadable sheet(s)."
            messagebox.showinfo("Template applied to all plots", summary)
        except Exception as exc:
            messagebox.showerror("Cannot apply template", str(exc))

    def _sync_settings_controls(self):
        for field in fields(PlotSettings):
            if field.name in self.plot_vars:
                value = getattr(self.settings, field.name)
                self.plot_vars[field.name].set("" if value is None else value)
        self.subtract_var.set(self.settings.subtract_background)
        self.normalize_var.set(self.settings.normalization)

    def save_project(self):
        if self.data is None or not self.collect_settings():
            return
        path = filedialog.asksaveasfilename(title="Save XPS Plot Studio project", defaultextension=".xpsplot", filetypes=[("XPS Plot Studio project", "*.xpsplot"), ("JSON", "*.json")])
        if path:
            Path(path).write_text(project_to_json(self.source_path, self.data.sheet_name, self.settings, self.traces, self.annotations), encoding="utf-8")
            self.status_var.set(f"Saved project {path}")

    def open_project(self):
        path = filedialog.askopenfilename(title="Open XPS Plot Studio project", filetypes=[("XPS Plot Studio project", "*.xpsplot *.json")])
        if not path:
            return
        try:
            payload = project_from_json(Path(path).read_text(encoding="utf-8"))
            source = payload["source_path"]
            if not Path(source).exists():
                source = filedialog.askopenfilename(title="Locate the workbook used by this project", filetypes=[("Excel workbook", "*.xlsx *.xlsm")])
                if not source:
                    return
            self.source_path = source
            self.sheet_states.clear()
            sheets = list_xlsx_sheets(source)
            self.sheet_combo["values"] = sheets
            self.sheet_var.set(payload["sheet_name"])
            self.file_var.set(source)
            self.data = load_xps_sheet(source, payload["sheet_name"])
            self.settings = PlotSettings(**payload.get("settings", {}))
            self.traces = [TraceStyle(**item) for item in payload.get("traces", [])]
            self.annotations = [AnnotationSpec(**item) for item in payload.get("annotations", [])]
            self._sync_settings_controls()
            self.refresh_trace_tree()
            self.refresh_annotation_tree()
            self.update_plot()
            self._remember_current_sheet()
            self.status_var.set(f"Opened project {path}")
        except Exception as exc:
            messagebox.showerror("Cannot open project", str(exc))

    def show_help(self):
        messagebox.showinfo(
            "Quick help",
            "1. Open an Avantage or XPS-master Excel workbook and choose a scientific sheet.\n"
            "2. XPS-master Summary sheets and model-comparison helper columns are ignored automatically.\n"
            "3. Double-click 'On' in the trace table to show/hide a curve. Accepted deconvoluted components are loaded.\n"
            "4. Fitted spectra open with a separate residual panel; toggle it under Plot & Axes.\n"
            "5. Select a trace to edit its color, thickness, opacity, fill, and gradient.\n"
            "6. Auto-label peaks, then drag labels directly on the plot.\n"
            "7. Export saves the visible plot exactly as previewed. Export All uses the same live canvas for each sheet.\n\n"
            "Use $...$ in labels for math text, e.g. $Fe^{3+}$. Templates contain reusable styling only.",
        )


def headless_export(workbook: str, sheet: str | None, output: str):
    import matplotlib
    matplotlib.use("Agg")
    sheets = list_xlsx_sheets(workbook)
    chosen = sheet or next((s for s in sheets if "scan" in s.lower()), sheets[0])
    data = load_xps_sheet(workbook, chosen)
    settings = PlotSettings(title=chosen.replace(" Scan", ""))
    traces = default_trace_styles(data)
    notes = auto_annotations(data, settings, traces)
    fig, _, _ = make_figure(data, settings, traces, notes)
    save_figure(fig, output, settings.dpi, settings.transparent_export)
    print(json.dumps({"output": str(Path(output).resolve()), "sheet": chosen, "points": len(data.energy), "components": len(data.components)}))


def main():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("workbook", nargs="?", help="Optional workbook to open")
    parser.add_argument("--render", metavar="WORKBOOK", help="Headless verification/export mode")
    parser.add_argument("--sheet", help="Sheet name for --render")
    parser.add_argument("--output", default="xps_figure.png", help="Output path for --render")
    args = parser.parse_args()
    if args.render:
        headless_export(args.render, args.sheet, args.output)
        return
    app = XPSPlotStudio(args.workbook)
    app.mainloop()


if __name__ == "__main__":
    main()
