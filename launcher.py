#!/usr/bin/env python
"""launcher.py — TUI launcher for GraphiXS training.
Usage:
    python launcher.py                # wizard, then run / submit
"""

import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.filters import completion_is_selected
from prompt_toolkit.key_binding import KeyBindings

# --------------------------------------------------------------------------------------
# Colour palette
# --------------------------------------------------------------------------------------
ACCENT = "bright_cyan"
DIM = "grey50"
GOOD = "bright_green"
WARN = "yellow"
ERR = "bright_red"
HEAD = "bold bright_white"

console = Console()

TRAIN_SCRIPT = "train.py"

BANNER = r"""
  ____                 _     _ __  ______
 / ___|_ __ __ _ _ __ | |__ (_)\ \/ / ___|
| |  _| '__/ _` | '_ \| '_ \| | \  /\___ \
| |_| | | | (_| | |_) | | | | | /  \ ___) |
 \____|_|  \__,_| .__/|_| |_|_|/_/\_\____/
                |_|
"""


# --------------------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    source_path: str = ""
    model_path: str = ""
    config_file: str = ""
    extra_args: str = ""
    start_checkpoint: str = ""
    clear_images: bool = False
    use_slurm: bool = False
    gpu_index: int = 0
    slurm_job_name: str = "gs_train"
    slurm_gpus: int = 1
    slurm_time: str = "24:00:00"
    slurm_partition: str = ""
    slurm_extra: str = ""


@dataclass
class TrainState:
    current_iter: int = 0
    total_iters: int = 30000
    loss: float = 0.0
    test_psnr: float = 0.0
    train_psnr: float = 0.0
    num_gaussians: int = 0
    start_time: float = field(default_factory=time.time)
    last_save: int = 0
    last_eval: int = 0
    saves: set = field(default_factory=set)
    planned_saves: set = field(default_factory=set)
    log_lines: list = field(default_factory=list)
    log_path: str = ""
    done: bool = False
    error: str = ""


# --------------------------------------------------------------------------------------
# GPU helpers (nvidia-smi)
# --------------------------------------------------------------------------------------
def detect_gpus():
    """Return a list of {index, name, vram} dicts; [] on any error."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"index": int(parts[0]), "name": parts[1], "vram": parts[2]})
        return gpus
    except Exception:
        return []


def poll_gpu_stats(index):
    """Single nvidia-smi query for one device. Returns {} on any error."""
    fields = ("utilization.gpu,memory.used,memory.total,temperature.gpu,"
              "power.draw,power.limit,clocks.current.graphics,clocks.current.memory,"
              "fan.speed")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}",
             "--format=csv,noheader,nounits", "-i", str(index)],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return {}
        line = out.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        keys = ["util", "vram_used", "vram_total", "temp", "pwr_draw",
                "pwr_limit", "clk_gr", "clk_mem", "fan"]
        stats = {}
        for k, v in zip(keys, parts):
            try:
                stats[k] = float(v)
            except ValueError:
                stats[k] = None
        return stats
    except Exception:
        return {}


# --------------------------------------------------------------------------------------
# Wizard
# --------------------------------------------------------------------------------------
def _banner_renderable():
    width = shutil.get_terminal_size().columns
    if width < 72:
        return Rule("GraphiXS — Launcher", style=ACCENT)
    return Text(BANNER, style=ACCENT)


class _SlashPathCompleter(Completer):
    """PathCompleter that appends '/' to directory completions, so accepting a
    folder lets the next Tab descend into it (zsh-style nested navigation)."""

    def __init__(self, only_directories=False):
        self._inner = PathCompleter(only_directories=only_directories, expanduser=True)

    def get_completions(self, document, complete_event):
        cur = document.text_before_cursor
        seen = set()

        def emit(text, start_position, **kw):
            key = (start_position, text)
            if key in seen:
                return None
            seen.add(key)
            return Completion(text, start_position=start_position, **kw)

        # Already-typed exact directory with no trailing slash: offer to descend.
        if cur and not cur.endswith("/") and os.path.isdir(os.path.expanduser(cur)):
            c = emit("/", 0, display=os.path.basename(cur.rstrip("/")) + "/")
            if c is not None:
                yield c
        for c in self._inner.get_completions(document, complete_event):
            # Reconstruct the full path this completion would produce (handles both
            # suffix-style and full-name completions via start_position).
            head = cur[:len(cur) + c.start_position]
            full = os.path.expanduser(head + c.text)
            text = c.text + "/" if (os.path.isdir(full) and not c.text.endswith("/")) else c.text
            out = emit(text, c.start_position, display=c.display, style=c.style)
            if out is not None:
                yield out


# Enter, while a completion is highlighted, confirms that completion (and closes
# the menu) instead of submitting the line — so Tab→Enter→Tab→Enter descends the
# directory tree. A final Enter (with no menu open) submits as usual.
_PATH_KB = KeyBindings()


@_PATH_KB.add("enter", filter=completion_is_selected)
def _(event):
    event.current_buffer.complete_state = None


def _read_path(prompt, default="", only_directories=False):
    """Read a path. On a TTY, Tab opens a completion menu navigable with the
    arrow keys (zsh-style) via prompt_toolkit; otherwise fall back to a plain
    rich prompt (e.g. when stdin is piped)."""
    if sys.stdin.isatty() and sys.stdout.isatty():
        completer = _SlashPathCompleter(only_directories=only_directories)
        text = PromptSession().prompt(
            f"{prompt}: ", completer=completer, key_bindings=_PATH_KB,
            complete_while_typing=False, default=default or "")
        return text.strip()
    val = Prompt.ask(f"[{ACCENT}]{prompt}[/]", default=default or "")
    return val.strip() if val else ""


_KW_ADJ = ["brave", "calm", "clever", "cosmic", "crimson", "dapper", "eager", "fuzzy",
           "gentle", "golden", "jolly", "lucid", "mellow", "nimble", "plucky", "quiet",
           "rapid", "rustic", "silver", "sly", "stellar", "sunny", "swift", "tidal",
           "velvet", "witty", "zesty", "amber", "azure", "bold"]
_KW_NOUN = ["otter", "falcon", "maple", "comet", "willow", "pixel", "quartz", "raven",
            "cedar", "lotus", "ember", "harbor", "meadow", "nimbus", "onyx", "panda",
            "reef", "sparrow", "tundra", "vortex", "walrus", "yak", "zephyr", "badger",
            "cobra", "dune", "fern", "glade", "heron", "ibis"]


def generate_keyword(base_dir="output", prefix=""):
    """Return a short, memorable, unique 'adjective-noun' keyword that does not
    collide with an existing '<prefix>_<keyword>' folder under base_dir."""
    for _ in range(50):
        kw = f"{random.choice(_KW_ADJ)}_{random.choice(_KW_NOUN)}"
        name = f"{prefix}_{kw}" if prefix else kw
        if not os.path.exists(os.path.join(base_dir, name)):
            return kw
    return f"{random.choice(_KW_ADJ)}_{random.choice(_KW_NOUN)}_{random.randint(1000, 9999)}"


def find_latest_checkpoint(model_path):
    """Return (path, iteration) of the highest-iteration chkpnt<N>.pth in
    model_path, or (None, 0) if there is none to resume from."""
    import glob
    best_path, best_iter = None, -1
    for p in glob.glob(os.path.join(model_path, "chkpnt*.pth")):
        m = re.match(r"chkpnt(\d+)\.pth$", os.path.basename(p))
        if m and int(m.group(1)) > best_iter:
            best_path, best_iter = p, int(m.group(1))
    return (best_path, best_iter) if best_path else (None, 0)


def _ask_path(prompt, check, default=None, only_directories=False):
    """Ask for a path; warn-and-confirm loop if `check(path)` is False."""
    while True:
        path = _read_path(prompt, default=default or "", only_directories=only_directories)
        if check(path):
            return path
        console.print(f"[{WARN}]Path not found:[/] {path}")
        if Confirm.ask(f"[{ACCENT}]Use anyway?[/]", default=False):
            return path


def wizard():
    console.clear()
    console.print(Align.center(_banner_renderable()))
    cfg = TrainConfig()

    # ---- Q1: Data source -------------------------------------------------------------
    console.print(Rule("1 / 5  ·  Data source", style=DIM))
    cfg.source_path = _ask_path("Source data path", os.path.isdir, only_directories=True)

    # ---- Q2: Config file -------------------------------------------------------------
    console.print(Rule("2 / 5  ·  Config file", style=DIM))
    cfg.config_file = _ask_path("Config file path", os.path.isfile)

    # ---- Q3: Output folder -----------------------------------------------------------
    console.print(Rule("3 / 5  ·  Output folder", style=DIM))
    src_name = os.path.basename(os.path.normpath(cfg.source_path)) or "data"
    cfg_name = os.path.splitext(os.path.basename(cfg.config_file))[0] or "config"
    prefix = f"{src_name}_{cfg_name}"
    default_name = f"{prefix}_{generate_keyword(prefix=prefix)}"
    folder = Prompt.ask(
        f"[{ACCENT}]Output folder name[/]", default=default_name).strip()
    if os.path.isabs(folder):
        cfg.model_path = folder
    else:
        cfg.model_path = os.path.join("output", folder)
    os.makedirs(os.path.join(cfg.model_path, "logs"), exist_ok=True)
    console.print(f"[{DIM}]Output → {cfg.model_path}[/]")

    ckpt, ckpt_iter = find_latest_checkpoint(cfg.model_path)
    if ckpt:
        console.print(f"[{WARN}]Existing checkpoint found:[/] "
                      f"{os.path.basename(ckpt)} [{DIM}](iter {ckpt_iter})[/]")
        if Confirm.ask(f"[{ACCENT}]Resume training from this checkpoint?[/]", default=True):
            cfg.start_checkpoint = ckpt
            console.print(f"[{GOOD}]→ Resuming from iter {ckpt_iter}.[/]")
            saved_cfg = os.path.join(cfg.model_path, "cfg_args.json")
            if os.path.isfile(saved_cfg):
                cfg.config_file = saved_cfg
                console.print(f"[{GOOD}]→ Reusing saved config "
                              f"{os.path.basename(saved_cfg)} from the output folder.[/]")
        else:
            console.print(f"[{DIM}]Starting fresh (existing checkpoints left untouched).[/]")

    # ---- Q4: Execution environment ---------------------------------------------------
    console.print(Rule("4 / 5  ·  Execution environment", style=DIM))
    sbatch = shutil.which("sbatch")
    if sbatch:
        console.print(f"[{GOOD}]✓ sbatch found[/] [{DIM}]({sbatch})[/]")
    else:
        console.print(f"[{DIM}]sbatch not found on PATH.[/]")
    mode = Prompt.ask(
        f"[{ACCENT}]Run directly or use SLURM to run?[/]",
        choices=["direct", "slurm"], default="slurm" if sbatch else "direct")
    cfg.use_slurm = (mode == "slurm")

    if cfg.use_slurm:
        cfg.slurm_job_name = Prompt.ask(f"[{ACCENT}]Job name[/]", default="gs_train")
        while True:
            cfg.slurm_gpus = IntPrompt.ask(f"[{ACCENT}]Number of GPUs[/]", default=1)
            if cfg.slurm_gpus >= 1:
                break
            console.print(f"[{WARN}]Must be ≥ 1.[/]")
        cfg.slurm_time = Prompt.ask(
            f"[{ACCENT}]Wall-clock time limit (HH:MM:SS)[/]", default="24:00:00")
        cfg.slurm_partition = Prompt.ask(
            f"[{ACCENT}]Partition / queue[/] [{DIM}](blank = cluster default)[/]",
            default="").strip()
        cfg.slurm_extra = Prompt.ask(
            f"[{ACCENT}]Extra #SBATCH flags[/] [{DIM}](blank = none)[/]",
            default="").strip()
    else:
        gpus = detect_gpus()
        if gpus:
            tbl = Table(box=None, pad_edge=False)
            tbl.add_column("idx", style=ACCENT, justify="right")
            tbl.add_column("name", style=HEAD)
            tbl.add_column("VRAM (MB)", style=DIM, justify="right")
            for g in gpus:
                tbl.add_row(str(g["index"]), g["name"], g["vram"])
            console.print(tbl)
        else:
            console.print(f"[{DIM}]No GPUs detected via nvidia-smi.[/]")
        cfg.gpu_index = IntPrompt.ask(f"[{ACCENT}]GPU index[/]", default=0)
        chosen = next((g for g in gpus if g["index"] == cfg.gpu_index), None)
        if chosen:
            console.print(f"[{GOOD}]→ Using GPU {cfg.gpu_index}: {chosen['name']}[/]")
        console.print(f"[{DIM}]After training, render (videos) + metrics run automatically.[/]")
        cfg.clear_images = Confirm.ask(
            f"[{ACCENT}]Clear rendered images after metrics (keep only videos)?[/]",
            default=False)

    # ---- Q5: Extra CLI parameters ----------------------------------------------------
    console.print(Rule("5 / 5  ·  Extra CLI parameters", style=DIM))
    console.print(f"[{DIM}]e.g. --iterations 30000 --checkpoint_iterations 7000 30000[/]")
    cfg.extra_args = Prompt.ask(
        f"[{ACCENT}]Extra arguments[/] [{DIM}](blank = none)[/]", default="").strip()

    return cfg


# --------------------------------------------------------------------------------------
# Summary panel
# --------------------------------------------------------------------------------------
def summary_panel(cfg):
    tbl = Table(box=None, pad_edge=False, show_header=False)
    tbl.add_column(style=DIM, justify="right")
    tbl.add_column(style=HEAD)

    if cfg.use_slurm:
        tbl.add_row("Mode", f"[{ACCENT}]HPC / Slurm (sbatch)[/]")
        tbl.add_row("Job name", cfg.slurm_job_name)
        tbl.add_row("GPUs", str(cfg.slurm_gpus))
        tbl.add_row("Time", cfg.slurm_time)
        tbl.add_row("Partition", cfg.slurm_partition or f"[{DIM}](default)[/]")
        if cfg.slurm_extra:
            tbl.add_row("Extra #SBATCH", cfg.slurm_extra)
    else:
        tbl.add_row("Mode", f"[{ACCENT}]Local GPU (device:{cfg.gpu_index})[/]")
        tbl.add_row("After training", "render (videos) + metrics"
                    + (", clear images" if cfg.clear_images else ""))

    tbl.add_row("Source", cfg.source_path)
    tbl.add_row("Output", cfg.model_path)
    tbl.add_row("Config", cfg.config_file)
    if cfg.start_checkpoint:
        tbl.add_row("Resume from", f"[{GOOD}]{os.path.basename(cfg.start_checkpoint)}[/]")
    if cfg.extra_args:
        tbl.add_row("Extra args", cfg.extra_args)

    return Panel(tbl, title="Run summary", border_style=ACCENT, padding=(1, 2))


# --------------------------------------------------------------------------------------
# Command builders
# --------------------------------------------------------------------------------------
def build_train_command(train_script, cfg):
    """Core python invocation as a single string (used inside sbatch --wrap)."""
    cmd = (f"python {train_script} -s {cfg.source_path} -m {cfg.model_path} "
           f"--config {cfg.config_file}")
    if cfg.start_checkpoint:
        cmd += f" --start_checkpoint {cfg.start_checkpoint}"
    if cfg.extra_args:
        cmd += f" {cfg.extra_args}"
    return cmd


def build_local_command(train_script, cfg):
    """Same as a token list for subprocess.Popen (CUDA_VISIBLE_DEVICES set via env)."""
    cmd = ["python", train_script,
           "-s", cfg.source_path,
           "-m", cfg.model_path,
           "--config", cfg.config_file]
    if cfg.start_checkpoint:
        cmd += ["--start_checkpoint", cfg.start_checkpoint]
    if cfg.extra_args:
        cmd += cfg.extra_args.split()
    return cmd


def build_sbatch_command(train_script, cfg):
    cmd = [
        "sbatch",
        f"--job-name={cfg.slurm_job_name}",
        f"--gres=gpu:{cfg.slurm_gpus}",
        f"--time={cfg.slurm_time}",
        f"--output={cfg.model_path}/logs/%x_%j.out",
        f"--error={cfg.model_path}/logs/%x_%j.err",
    ]
    if cfg.slurm_partition:
        cmd.append(f"--partition={cfg.slurm_partition}")
    if cfg.slurm_extra:
        cmd += cfg.slurm_extra.split()
    cmd.append(f'--wrap={build_train_command(train_script, cfg)}')
    return cmd


# --------------------------------------------------------------------------------------
# Slurm submission
# --------------------------------------------------------------------------------------
def submit_slurm(train_script, cfg):
    cmd = build_sbatch_command(train_script, cfg)

    # Pretty-print, each flag on its own line.
    pretty = Text()
    for i, tok in enumerate(cmd):
        cont = " \\" if i < len(cmd) - 1 else ""
        prefix = "" if i == 0 else "    "
        pretty.append(f"{prefix}{tok}{cont}\n", style=DIM)
    console.print(Panel(pretty, title="sbatch command", border_style=DIM))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(Panel(result.stderr.strip() or "sbatch failed.",
                            title="Submission error", border_style=ERR))
        sys.exit(1)

    m = re.search(r"(\d+)", result.stdout)
    job_id = m.group(1) if m else "?"

    info = Table(box=None, show_header=False, pad_edge=False)
    info.add_column(style=DIM, justify="right")
    info.add_column(style=HEAD)
    info.add_row("Job ID", job_id)
    info.add_row("Job name", cfg.slurm_job_name)
    info.add_row("Stdout log", f"{cfg.model_path}/logs/{cfg.slurm_job_name}_{job_id}.out")
    info.add_row("Stderr log", f"{cfg.model_path}/logs/{cfg.slurm_job_name}_{job_id}.err")
    console.print(Panel(info, title="Submitted", border_style=GOOD, padding=(1, 2)))

    track = Text(f"squeue --job {job_id}", style=ACCENT)
    console.print(Panel(track, title="Track", border_style=ACCENT, box=box.MINIMAL))


# --------------------------------------------------------------------------------------
# Log parsing
# --------------------------------------------------------------------------------------
_RE_ITER = re.compile(r"\[ITER\s+(\d+)(?:/(\d+))?\]")
_RE_LOSS = re.compile(r"loss=([0-9.eE+\-]+)")
_RE_GAUSS = re.compile(r"gaussians=(\d+)")
_RE_TEST_PSNR = re.compile(r"best test PSNR:\s*([0-9.]+)")
_RE_TRAIN_PSNR = re.compile(r"best train PSNR:\s*([0-9.]+)")
_RE_SAVE = re.compile(r"[Ss]aving|[Cc]heckpoint")
_RE_EVAL = re.compile(r"[Ee]valuating")
_RE_SAVE_ITERS = re.compile(r"\[SAVE_ITERS\]\s*([\d\s]*)")


def parse_line(line, state):
    m = _RE_SAVE_ITERS.search(line)
    if m:
        state.planned_saves = {int(x) for x in m.group(1).split()}
        return
    m = _RE_ITER.search(line)
    if m:
        state.current_iter = int(m.group(1))
        if m.group(2):
            state.total_iters = int(m.group(2))
    m = _RE_LOSS.search(line)
    if m:
        try:
            state.loss = float(m.group(1))
        except ValueError:
            pass
    m = _RE_GAUSS.search(line)
    if m:
        state.num_gaussians = int(m.group(1))
    m = _RE_TEST_PSNR.search(line)
    if m:
        try:
            state.test_psnr = float(m.group(1))
        except ValueError:
            pass
    m = _RE_TRAIN_PSNR.search(line)
    if m:
        try:
            state.train_psnr = float(m.group(1))
        except ValueError:
            pass
    if _RE_EVAL.search(line):
        state.last_eval = state.current_iter
    if _RE_SAVE.search(line):
        state.last_save = state.current_iter
        if state.current_iter:
            state.saves.add(state.current_iter)


def parse_save_iterations(extra_args):
    """Scan extra_args for --checkpoint_iterations / --test_iterations integers."""
    marks = set()
    tokens = extra_args.split()
    i = 0
    while i < len(tokens):
        if tokens[i] in ("--checkpoint_iterations", "--test_iterations", "--save_iterations"):
            j = i + 1
            while j < len(tokens) and tokens[j].lstrip("-").isdigit():
                marks.add(int(tokens[j]))
                j += 1
            i = j
        else:
            i += 1
    return sorted(marks)


def parse_total_iters(extra_args, default=30000):
    tokens = extra_args.split()
    for i, t in enumerate(tokens):
        if t == "--iterations" and i + 1 < len(tokens):
            try:
                return int(tokens[i + 1])
            except ValueError:
                return default
    return default


# --------------------------------------------------------------------------------------
# Helpers for the dashboard
# --------------------------------------------------------------------------------------
def fmt_hms(seconds):
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def loss_colour(loss):
    if loss < 0.01:
        return GOOD
    if loss < 0.05:
        return ACCENT
    if loss < 0.1:
        return WARN
    return ERR


def gauge(label, value, vmax, unit="", yellow=None, red=None, width=20):
    """A single coloured gauge bar. Returns a Text."""
    t = Text()
    t.append(f"{label:<6}", style=DIM)
    if value is None or vmax in (None, 0):
        t.append("n/a", style=DIM)
        return t
    frac = max(0.0, min(value / vmax, 1.0))
    filled = int(frac * width)
    colour = ACCENT
    if red is not None and value >= red:
        colour = ERR
    elif yellow is not None and value >= yellow:
        colour = WARN
    t.append("━" * filled, style=colour)
    t.append("╌" * (width - filled), style=DIM)
    t.append(f" {value:5.1f}{unit}", style=HEAD)
    return t


# --------------------------------------------------------------------------------------
# Dashboard render functions
# --------------------------------------------------------------------------------------
def render_header(cfg, state):
    elapsed = time.time() - state.start_time
    it = state.current_iter
    total = max(state.total_iters, 1)
    pct = 100.0 * it / total
    speed = it / elapsed if elapsed > 0 else 0.0
    remaining = (total - it) / speed if speed > 0 else 0
    gpu_name = f"GPU:{cfg.gpu_index}"

    exp_name = os.path.basename(os.path.normpath(cfg.model_path))
    t = Text()
    t.append(exp_name, style=HEAD)
    t.append("  ·  ", style=DIM)
    t.append(gpu_name, style=ACCENT)
    t.append("  ·  elapsed ", style=DIM)
    t.append(fmt_hms(elapsed), style=HEAD)
    t.append("  ·  ETA ", style=DIM)
    t.append(fmt_hms(remaining), style=HEAD)
    t.append("  ·  ", style=DIM)
    t.append(f"{speed:.1f} it/s", style=ACCENT)
    t.append("  ·  ", style=DIM)
    t.append(f"{pct:5.1f}%", style=GOOD)
    return Panel(Align.center(t), border_style=ACCENT)


class _ProgressBar:
    """A single-line progress bar that sizes itself to the width it is actually
    given at render time (so it fits its panel, not the whole terminal). The
    iteration count + percentage sit on the right, and checkpoint markers are
    overlaid directly on the bar: green once a save has happened at that
    iteration, else yellow (planned / upcoming)."""

    def __init__(self, state, save_marks):
        self.state = state
        self.save_marks = save_marks

    def __rich_console__(self, console, options):
        state = self.state
        total = max(state.total_iters, 1)
        pct = 100.0 * min(state.current_iter / total, 1.0)
        label = f"  {state.current_iter}/{total}  {pct:5.1f}%"
        avail = max(options.max_width, len(label) + 11)
        width = max(avail - len(label), 10)
        frac = max(0.0, min(state.current_iter / total, 1.0))
        filled = int(frac * width)

        cells = ["━"] * filled + ["╌"] * (width - filled)
        styles = [ACCENT] * filled + [DIM] * (width - filled)
        for mk in set(self.save_marks) | state.planned_saves | state.saves:
            pos = min(int(width * mk / total), width - 1)
            if 0 <= pos < width:
                cells[pos] = "┃"
                styles[pos] = GOOD if mk in state.saves else WARN

        bar = Text()
        for ch, st in zip(cells, styles):
            bar.append(ch, style=st)
        bar.append(label, style=HEAD)
        yield bar


def render_progress(state, save_marks):
    return Panel(_ProgressBar(state, save_marks), title="Progress",
                 border_style=ACCENT, padding=(0, 1))


def render_metrics(state):
    tbl = Table(box=None, show_header=False, pad_edge=False, expand=True)
    tbl.add_column(style=DIM, justify="right")
    tbl.add_column(style=HEAD)
    saved = f"[{WARN}]{state.last_save}[/]" if state.last_save else f"[{DIM}]—[/]"
    evaled = f"[{ACCENT}]{state.last_eval}[/]" if state.last_eval else f"[{DIM}]—[/]"
    tbl.add_row("Iteration", f"{state.current_iter} / {state.total_iters}")
    tbl.add_row("Loss", f"[{loss_colour(state.loss)}]{state.loss:.6f}[/]")
    tbl.add_row("Number of Components", f"{state.num_gaussians:,}")
    tbl.add_row("Last saved", saved)
    tbl.add_row("Last evaluated", evaled)
    tbl.add_row("Last Train PSNR",
                f"[{GOOD}]{state.train_psnr:.2f} dB[/]" if state.train_psnr else f"[{DIM}]—[/]")
    tbl.add_row("Last Test PSNR",
                f"[{GOOD}]{state.test_psnr:.2f} dB[/]" if state.test_psnr else f"[{DIM}]—[/]")
    return Panel(tbl, title="Metrics", border_style=ACCENT, padding=(0, 1))


def render_gpu(stats, peak):
    if not stats:
        return Panel(Align.center(Text("nvidia-smi unavailable", style=DIM)),
                     title="GPU", border_style=DIM)
    vram_used = stats.get("vram_used") or 0
    vram_total = stats.get("vram_total") or 1
    vram_pct = 100.0 * vram_used / vram_total
    pwr_draw = stats.get("pwr_draw")
    pwr_limit = stats.get("pwr_limit") or 1

    rows = [
        gauge("VRAM", vram_pct, 100, "%",
              yellow=75, red=90),
        gauge("Util", stats.get("util"), 100, "%"),
        gauge("Temp", stats.get("temp"), 100, "°", yellow=70, red=82),
        gauge("Power", (100.0 * pwr_draw / pwr_limit) if pwr_draw is not None else None,
              100, "%", yellow=90),
        gauge("Fan", stats.get("fan"), 100, "%"),
    ]
    extra = Table(box=None, show_header=False, pad_edge=False)
    extra.add_column(style=DIM, justify="right")
    extra.add_column(style=HEAD)
    extra.add_row("VRAM", f"{vram_used:.0f}/{vram_total:.0f} MB")
    extra.add_row("Peak VRAM", f"{peak.get('vram', 0):.0f} MB")
    extra.add_row("Temp", f"{stats.get('temp') or 0:.0f} °C")
    extra.add_row("Peak temp", f"{peak.get('temp', 0):.0f} °C")

    return Panel(Group(*rows, Rule(style=DIM), extra),
                 title="GPU monitor", border_style=ACCENT, padding=(0, 1))


def render_log(state):
    size = shutil.get_terminal_size()
    n = max(size.lines - 32, 4)
    lines = state.log_lines[-n:]
    body = Text()
    for ln in lines:
        body.append(ln.rstrip() + "\n", style=DIM)
    return Panel(body, title="Log", border_style=DIM, padding=(0, 1))


def render_sidebar(cfg):
    tbl = Table(box=None, show_header=False, pad_edge=False)
    tbl.add_column(style=DIM, justify="right")
    tbl.add_column(style=HEAD, overflow="fold")
    tbl.add_row("Source", cfg.source_path)
    tbl.add_row("Output", cfg.model_path)
    tbl.add_row("Config", cfg.config_file)
    if cfg.extra_args:
        ea = cfg.extra_args if len(cfg.extra_args) <= 60 else cfg.extra_args[:57] + "..."
        tbl.add_row("Extra", ea)
    return Panel(tbl, title="Run", border_style=ACCENT, padding=(0, 1))


def render_footer(state):
    if state.error:
        t = Text(state.error, style=ERR)
    elif state.done:
        t = Text("Training complete.", style=GOOD)
    else:
        t = Text("Training… press Ctrl+C to stop.", style=DIM)
    return Panel(Align.center(t), border_style=DIM)


def build_layout(cfg, state, stats, peak, save_marks):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_row(
        Layout(name="left", ratio=3),
        Layout(name="right", ratio=2),
    )
    layout["left"].split_column(
        Layout(name="progress", size=3),
        Layout(name="metrics", size=9),
        Layout(name="log"),
    )
    layout["right"].split_column(
        Layout(name="gpu", ratio=2),
        Layout(name="sidebar", ratio=3),
    )

    layout["header"].update(render_header(cfg, state))
    layout["progress"].update(render_progress(state, save_marks))
    layout["metrics"].update(render_metrics(state))
    layout["log"].update(render_log(state))
    layout["gpu"].update(render_gpu(stats, peak))
    layout["sidebar"].update(render_sidebar(cfg))
    layout["footer"].update(render_footer(state))
    return layout


# --------------------------------------------------------------------------------------
# Local training dashboard
# --------------------------------------------------------------------------------------
def run_training(train_script, cfg):
    state = TrainState()
    state.total_iters = parse_total_iters(cfg.extra_args)
    save_marks = parse_save_iterations(cfg.extra_args)
    gpu_stats = {}
    peak = {"vram": 0.0, "temp": 0.0}

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_index)

    cmd = build_local_command(train_script, cfg)

    # Persist all stdout to a timestamped log file so failed runs can be inspected.
    log_dir = os.path.join(cfg.model_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    state.log_path = os.path.join(log_dir, f"train_{datetime.now():%Y%m%d_%H%M%S}.log")
    log_file = open(state.log_path, "w", buffering=1)
    log_file.write(f"# CUDA_VISIBLE_DEVICES={cfg.gpu_index} " + " ".join(cmd) + "\n")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)

    def reader():
        try:
            for line in proc.stdout:
                log_file.write(line)
                parse_line(line, state)
                state.log_lines.append(line.rstrip())
                if len(state.log_lines) > 500:
                    del state.log_lines[:-400]
        finally:
            proc.wait()
            log_file.flush()
            log_file.close()
        state.done = True
        if proc.returncode not in (0, None) and not state.error:
            state.error = f"Process exited with code {proc.returncode}"

    def gpu_poller():
        while not state.done:
            stats = poll_gpu_stats(cfg.gpu_index)
            if stats:
                gpu_stats.clear()
                gpu_stats.update(stats)
                if stats.get("vram_used"):
                    peak["vram"] = max(peak["vram"], stats["vram_used"])
                if stats.get("temp"):
                    peak["temp"] = max(peak["temp"], stats["temp"])
            time.sleep(2)

    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=gpu_poller, daemon=True).start()

    try:
        with Live(build_layout(cfg, state, gpu_stats, peak, save_marks),
                  console=console, screen=True, refresh_per_second=4) as live:
            while not state.done:
                live.update(build_layout(cfg, state, gpu_stats, peak, save_marks))
                time.sleep(0.25)
            live.update(build_layout(cfg, state, gpu_stats, peak, save_marks))
    except KeyboardInterrupt:
        proc.terminate()
        state.error = "Interrupted by user"
        state.done = True
        with Live(build_layout(cfg, state, gpu_stats, peak, save_marks),
                  console=console, screen=True, refresh_per_second=4):
            time.sleep(1)

    post_run_summary(cfg, state, peak)

    # On a successful run, render + compute metrics for the trained scene.
    if state.done and not state.error:
        run_render_and_metrics(cfg)


def _run_step(title, cmd, env):
    """Run one post-training subprocess (inherited stdout), return True on success."""
    console.rule(f"[{ACCENT}]{title}[/]")
    console.print(f"[{DIM}]$ {' '.join(cmd)}[/]")
    try:
        rc = subprocess.run(cmd, env=env).returncode
    except Exception as e:
        console.print(f"[{ERR}]Failed to run {title}: {e}[/]")
        return False
    if rc != 0:
        console.print(f"[{ERR}]{title} exited with code {rc}.[/]")
        return False
    return True


def run_render_and_metrics(cfg):
    """After training, render the largest-iteration and best_test checkpoints,
    then compute metrics (mirrors shell_scripts/render_and_metrics_single.sh)."""
    model_path = cfg.model_path
    config = os.path.join(model_path, "cfg_args.json")
    if not os.path.isfile(config):
        config = cfg.config_file
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_index)

    _, latest_iter = find_latest_checkpoint(model_path)
    best_test = os.path.join(model_path, "chkpnt_best_test.pth")

    console.print(Panel(f"Training complete — rendering & computing metrics for "
                        f"[{HEAD}]{os.path.basename(os.path.normpath(model_path))}[/]",
                        border_style=ACCENT, padding=(0, 2)))

    ok = True
    if latest_iter:
        ok &= _run_step(
            f"Render · iteration {latest_iter}",
            ["python", "render.py", "-m", model_path, "--config", config,
             "--render-videos", "--iteration", str(latest_iter), "--skip_train"], env)
    else:
        console.print(f"[{WARN}]No numbered checkpoint found — skipping latest-iteration render.[/]")

    if os.path.isfile(best_test):
        ok &= _run_step(
            "Render · best_test",
            ["python", "render.py", "-m", model_path, "--config", config,
             "--render-videos", "--iteration", "_best_test", "--skip_train"], env)
    else:
        console.print(f"[{WARN}]No chkpnt_best_test.pth — skipping best_test render.[/]")

    ok &= _run_step("Metrics",
                    ["python", "metrics.py", "-m", model_path, "--skip_train"], env)

    # Optionally clear rendered images, keeping the rendered videos only.
    if cfg.clear_images:
        removed = 0
        for sub in ("test", "train", "imgs"):
            d = os.path.join(model_path, sub)
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        try:
                            os.remove(os.path.join(root, f))
                            removed += 1
                        except OSError:
                            pass
        console.print(f"[{DIM}]Cleared {removed} rendered image(s); videos kept.[/]")

    border = GOOD if ok else ERR
    msg = ("Render + metrics complete." if ok
           else "Render/metrics finished with errors — see output above.")
    console.print(Panel(msg, border_style=border, padding=(0, 2)))


def post_run_summary(cfg, state, peak):
    interrupted = state.error == "Interrupted by user"
    success = state.done and not state.error
    border = GOOD if (success or interrupted) else ERR
    if success:
        status = f"[{GOOD}]Complete[/]"
    elif interrupted:
        status = f"[{WARN}]Interrupted[/]"
    else:
        status = f"[{ERR}]Failed[/]"

    tbl = Table(box=None, show_header=False, pad_edge=False)
    tbl.add_column(style=DIM, justify="right")
    tbl.add_column(style=HEAD)
    tbl.add_row("Status", status)
    tbl.add_row("Duration", fmt_hms(time.time() - state.start_time))
    tbl.add_row("Final iter", f"{state.current_iter} / {state.total_iters}")
    tbl.add_row("Train PSNR", f"{state.train_psnr:.2f} dB" if state.train_psnr else "—")
    tbl.add_row("Test PSNR", f"{state.test_psnr:.2f} dB" if state.test_psnr else "—")
    tbl.add_row("Final loss", f"{state.loss:.6f}")
    tbl.add_row("Number of Components", f"{state.num_gaussians:,}")
    tbl.add_row("Peak VRAM", f"{peak.get('vram', 0):.0f} MB")
    tbl.add_row("Peak temp", f"{peak.get('temp', 0):.0f} °C")
    tbl.add_row("Output", cfg.model_path)
    if state.log_path:
        tbl.add_row("Log file", state.log_path)
    if state.error and not interrupted:
        tbl.add_row("Error", f"[{ERR}]{state.error}[/]")
    console.print(Panel(tbl, title="Run finished", border_style=border, padding=(1, 2)))


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def main():
    cfg = wizard()
    console.print(summary_panel(cfg))

    if not Confirm.ask(f"[{ACCENT}]Launch now?[/]", default=True):
        console.print(f"[{DIM}]Aborted.[/]")
        return

    if cfg.use_slurm:
        submit_slurm(TRAIN_SCRIPT, cfg)
    else:
        run_training(TRAIN_SCRIPT, cfg)


if __name__ == "__main__":
    main()
