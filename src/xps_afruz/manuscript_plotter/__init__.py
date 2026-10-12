"""Publication plotting tools for Avantage and XPS-master Excel workbooks."""

from .core import (
    AnnotationSpec,
    PlotSettings,
    TraceStyle,
    XPSData,
    default_trace_styles,
    list_xlsx_sheets,
    load_xps_sheet,
    make_figure,
)

__all__ = [
    "AnnotationSpec",
    "PlotSettings",
    "TraceStyle",
    "XPSData",
    "default_trace_styles",
    "list_xlsx_sheets",
    "load_xps_sheet",
    "make_figure",
]
