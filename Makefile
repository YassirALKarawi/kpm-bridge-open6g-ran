.PHONY: test smoke fetch benchmark tables figures assets references paper claims audit clean

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v

smoke:
	mkdir -p reproducibility/outputs
	PYTHONPATH=src python3 scripts/run_smoke.py --output reproducibility/outputs/smoke_summary.json

fetch:
	python3 scripts/fetch_colosseum_subset.py

benchmark:
	PYTHONPATH=src python3 scripts/run_benchmark.py

tables:
	python3 scripts/make_latex_results.py

figures:
	MPLCONFIGDIR=.mplconfig python3 scripts/make_figures.py

assets: tables figures

references:
	python3 scripts/audit_references.py

paper:
	latexmk -cd -pdf -interaction=nonstopmode -halt-on-error manuscript/main.tex

claims:
	python3 scripts/audit_claims.py

audit: test references paper claims

clean:
	latexmk -cd -c manuscript/main.tex
