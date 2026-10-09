"""Terminal report — the disagreement, rendered."""
from __future__ import annotations

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

SEV_COLOR = {"critical": RED, "warning": YELLOW, "info": CYAN}


def render_verdict(label: str, extraction, verdict) -> str:
    lines = []
    head_color = GREEN if verdict.reconciled else RED
    lines.append(f"\n{BOLD}{'=' * 64}{RESET}")
    lines.append(f"{BOLD}{label}{RESET}")
    lines.append(f"{'=' * 64}")
    lines.append(f"  stated total : ₹{verdict.stated_total if verdict.stated_total is not None else '—'}")
    lines.append(f"  naive sum    : ₹{verdict.naive_total:,.2f}  "
                 f"{DIM}(what you get by trusting the model's JSON blindly){RESET}")
    lines.append(f"  adjudicated  : ₹{verdict.computed_total:,.2f}  "
                 f"{DIM}(charge rows only — the repair){RESET}")
    if verdict.naive_delta is not None and abs(verdict.naive_delta) > 0.005:
        lines.append(f"  blind error  : ₹{verdict.naive_delta:+,.2f}")
    if verdict.delta is not None:
        lines.append(f"  repaired Δ   : ₹{verdict.delta:+,.2f}")
    lines.append("")
    if not verdict.findings:
        lines.append(f"  {GREEN}no findings — the model's numbers reconcile{RESET}")
    for f in verdict.findings:
        c = SEV_COLOR.get(f.severity, "")
        lines.append(f"  {c}[{f.severity.upper()}]{RESET} {BOLD}{f.code}{RESET}")
        lines.append(f"      {f.detail}")
    lines.append("")
    lines.append(f"  {head_color}{BOLD}{verdict.summary()}{RESET}")
    return "\n".join(lines)
