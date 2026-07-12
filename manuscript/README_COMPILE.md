# Compile the manuscript

From this directory run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The source uses `IEEEtran` in 10-point journal mode and BibTeX with
`IEEEtran.bst`. Generated tables and numerical macros are already included
under `generated/`; the nine publication figures are vector PDF files under
`figures/`.

The audited output is 13 US-letter pages. A normal compilation may report
underfull box messages caused by IEEE column balancing; it must not report an
overfull box, missing citation, or unresolved cross-reference.
