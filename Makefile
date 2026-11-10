PY=.venv/bin/python
setup:
	/opt/homebrew/bin/python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
test:
	$(PY) -m pytest -q tests
outcomes:
	PYTHONPATH=. $(PY) scripts/run_strategies.py
frontier:
	PYTHONPATH=. $(PY) scripts/make_frontier.py
