"""Format the committed validation JSON files into the tables the pages show.

Every number on ``validation.md`` and ``reference/performance.md`` is read from
``_static/validation/*.json`` by these helpers, so a page cannot drift from the
files the scripts wrote.
"""

import json
from pathlib import Path


def static_dir():
    """Locate ``_static/validation`` from the page's working directory."""
    for base in (Path("."), Path("docs"), Path(__file__).resolve().parent.parent):
        if (base / "_static" / "validation").is_dir():
            return base / "_static" / "validation"
    raise FileNotFoundError("_static/validation")


def load(name):
    """Read one JSON file written by a script under ``docs/scripts``."""
    return json.loads((static_dir() / name).read_text())


def sci(value, digits=2):
    """Format a number as inline math ``a x 10^b``; zero stays 0."""
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def bin_tables(result, method):
    """Markdown tables, one per radius bin, of the three metrics for ``method``."""
    names = {
        "local_peak": "error over the local peak",
        "pixel_brightest": "error over each pixel's brightest value",
        "photometric": "core flux error (aperture radius 0.7)",
    }
    out = []
    for entry in result["bins"]:
        upper = "" if entry["r_max"] is None else f" to {entry['r_max']:.2f}"
        out.append(
            f"**{entry['name']}** (source radius {entry['r_min']:.2f}{upper} "
            f"$\\lambda_0/D$, {entry['n']} positions)\n"
        )
        out.append("| Metric | Median | 95th percentile | Worst |")
        out.append("|---|---|---|---|")
        for key, label in names.items():
            row = entry[method][key]
            cells = [sci(row[k]) for k in ("median", "p95", "worst")]
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)
