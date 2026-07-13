# Final JSAC validation — 13 July 2026

## Submission artifact

- Manuscript: `paper/KPM-Bridge_Manuscript.pdf`
- Format: IEEEtran journal mode, 10 point, two columns, US Letter
- Length: 13 pages including references
- PDF SHA-256: `b038ac9d2ddd7e214b0219220b39f528be0690d595022ea2bd85f71ffdd5fd5b`
- Source SHA-256 (`manuscript/main.tex`):
  `a820e30deaa976330be327f7e946eee0a3fbc3ced3e8374f22f9e33832baf13d`
- Bibliography SHA-256 (`manuscript/references.bib`):
  `7af020b4fbeeef7687c85cfb707960f79bfa377ba3882d6a2b80b9efd9d56f2d`

## Automated evidence

- `make test`: 18/18 deterministic tests passed.
- `make references`: 40/40 DOI records verified.
- `make claims`: 118/118 manuscript checks passed.
- `make paper`: 13-page PDF produced without overfull boxes, unresolved
  citations, or unresolved cross-references.
- All PDF fonts are embedded and subset.

## Structural audit

- 15 independently numbered figures: three architecture illustrations and 12
  result figures; no subfigures.
- 10 tables, each called out in the text.
- Two complete algorithms with inputs, ordered steps, outputs, and equation
  links.
- Five stated theoretical results with assumptions and proofs.
- 40 references in first-citation order. All have exact DOI links. The three
  directly relevant author publications occur at separated positions [21],
  [37], and [40].
- The manuscript contains the verified repository URL in a concise,
  standalone reproducibility section:
  `https://github.com/YassirALKarawi/kpm-bridge-open6g-ran`.

## Interpretation guardrails

The reported evidence uses a hash-pinned public ColO-RAN subset and controlled
implementation profiles rather than commercial multi-vendor traces. Confidence
interval overlap is reported where it prevents a superiority claim. Under the
deliberate P4 shift, exposure reduction is attributed mainly to abstention;
post-shift conditional error remains a stated limitation. These boundaries are
preserved in the abstract, results, discussion, conclusion, and cover letter.

The final PDF was also inspected page by page for clipping, overlap, unreadable
text, and unbalanced references. No blocking layout defect was found.
