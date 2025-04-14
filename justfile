set windows-powershell := true

# set up python environment and install requirements
setup: create-env pip-req

[unix]
create-env:
	python3 -m venv {{justfile_directory()}}/.venv
[windows]
create-env:
	python -m venv {{justfile_directory()}}\.venv

[unix]
pip-req:
	{{justfile_directory()}}/.venv/bin/pip install -r {{justfile_directory()}}/requirements.txt
[windows]
pip-req:
	{{justfile_directory()}}\.venv\Scripts\pip install -r {{justfile_directory()}}\requirements.txt

[unix]
run *file_and_args:
	cd {{invocation_directory()}}; {{justfile_directory()}}/.venv/bin/python3 {{file_and_args}}
[windows]
run *file_and_args:
	cd {{invocation_directory()}}; {{justfile_directory()}}\.venv\Scripts\python {{file_and_args}}

metacentrum-port address:
	ssh -L 8308:127.0.0.1:8308 -t {{address}}