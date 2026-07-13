# Compile the manuscript

From this directory run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The source uses `IEEEtran` in 10-point journal mode and BibTeX with
`IEEEtran.bst`. Generated tables and numerical macros are already included
under `generated/`. The three supplied architecture illustrations and twelve
independent vector result figures used by the manuscript are stored under
`figures/`; every included figure has its own number and caption, with no
subfigures. The additional model-size plot is retained as an audited repository
asset but is not included in the 13-page manuscript.

The audited output is 13 US-letter pages. A normal compilation may report
underfull box messages caused by IEEE column balancing; it must not report an
overfull box, missing citation, or unresolved cross-reference.
