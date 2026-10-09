"""Markdown -> LaTeX source (.tex) -> PDF, via pandoc + LuaLaTeX.

usage:
    python3 tools/md2pdf/md2pdf.py notes/scaling/memory-compute-sharding.md out.pdf [PATH_TO_PANDOC]

Writes OUT.tex next to OUT.pdf, then compiles that .tex, so the committed .tex is exactly the PDF's source.
Needs pandoc (brew install pandoc) and a TeX install with lualatex; texmf/ adds fvextra, lualatex-math and
xurl (missing from BasicTeX). To recompile a .tex without pandoc, run from the .tex file's folder (twice):
    TEXINPUTS=<repo>/tools/md2pdf/texmf//: LUAINPUTS=<repo>/tools/md2pdf/texmf//: lualatex FILE.tex
"""
import os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

here = Path(__file__).resolve().parent
src, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
tex = dst.with_suffix(".tex")
pandoc = sys.argv[3] if len(sys.argv) > 3 else "pandoc"
multi_h1 = len(re.findall(r"^# ", re.sub(r"```.*?```", "", src.read_text(), flags=re.S), flags=re.M)) > 1

# 1. Markdown -> standalone .tex; run pandoc from the .md's folder so image paths resolve the same way
cmd = [pandoc, src.name, "-s", "-t", "latex", "-o", str(tex), f"--lua-filter={here / 'filter.lua'}",
       f"--include-in-header={here / 'header.tex'}",
       "-V", "geometry:margin=2.2cm", "-V", "fontsize=11pt", "-V", "colorlinks=true", "-V", "linkcolor=blue!50!black",
       "-V", "urlcolor=blue!50!black", "-V", "documentclass=article"]
if multi_h1:
    cmd += ["--toc", "--toc-depth=2"]     # several "# Part ..." headings: add a table of contents
subprocess.run(cmd, check=True, cwd=src.parent)

# 2. .tex -> PDF, compiled from the .md's folder (where the image paths point); two passes for the contents
extra = f"{here / 'texmf'}//:"   # locally added packages; the trailing ":" keeps the default search path
env = dict(os.environ, PATH="/Library/TeX/texbin:" + os.environ["PATH"], TEXINPUTS=extra, LUAINPUTS=extra)
with tempfile.TemporaryDirectory() as build:
    for _ in range(2):
        r = subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={build}",
                            str(tex)], cwd=src.parent, env=env, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stdout[-3000:])
    shutil.copyfile(Path(build) / tex.with_suffix(".pdf").name, dst)
    log = (Path(build) / tex.with_suffix(".log").name).read_text(errors="replace")
missing = sorted(set(re.findall(r"Missing character: There is no (\S+)", log)))
print(f"{tex.name} + {dst.name}: {dst.stat().st_size / 1e6:.1f} MB" + (f"; missing glyphs: {missing}" if missing else ""))
