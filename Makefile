.PHONY: test smoke paper clean

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'

smoke:
	mkdir -p reproducibility/outputs
	PYTHONPATH=src python3 scripts/run_smoke.py --output reproducibility/outputs/smoke_summary.json

paper:
	cd manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	cd manuscript && latexmk -c
