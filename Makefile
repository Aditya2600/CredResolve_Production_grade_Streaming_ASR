PYTHON ?= python
MANIFEST ?= hold.jsonl
MODE ?= active

.PHONY: eval

eval:
	$(PYTHON) -m worker.eval.run_eval --manifest $(MANIFEST) --mode $(MODE)
