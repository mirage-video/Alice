.PHONY: format

format:
	isort generate.py alice
	yapf -i -r *.py generate.py alice
