# Compile the manuscript

From this directory run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The source uses `IEEEtran` in 10-point journal mode and BibTeX with
`IEEEtran.bst`. Generated tables and numerical macros are already included
under `generated/`. The three supplied architecture illustrations and thirteen
independent vector result figures are stored under `figures/`; every figure has
its own number and caption, with no subfigures.

The audited output is 13 US-letter pages. A normal compilation may report
underfull box messages caused by IEEE column balancing; it must not report an
overfull box, missing citation, or unresolved cross-reference.
