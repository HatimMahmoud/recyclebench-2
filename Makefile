.PHONY: test run validate pilot audit judge prereg figures clean
export PYTHONPATH := src

test:      ; python3 -m pytest tests/ -q
validate:  ; python3 -m recyclebench validate
pilot:     ; python3 -m recyclebench pilot
run:       ; python3 -m recyclebench run
audit:     ; python3 -m recyclebench audit-export --n 300
judge:     ; python3 -m recyclebench judge-export --n-bundles 60
prereg:    ; python3 -m recyclebench prereg
verify:    ; python3 -m recyclebench prereg-verify
figures:   ; python3 -m recyclebench figures
pminus:    ; @echo 'usage: python3 -m recyclebench pilot-pminus --old X --new Y --behaviours F.csv'
clean:     ; rm -rf results audit __pycache__ src/recyclebench/__pycache__ tests/__pycache__ .pytest_cache
