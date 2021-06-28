@echo off

SET 	match=re.match(r'^([a-zA-Z_-]+):.*?

IF /I "%1"==".ONESHELL" GOTO .ONESHELL
IF /I "%1"=="SHELL " GOTO SHELL 
IF /I "%1"=="PACKAGE_NAME " GOTO PACKAGE_NAME 
IF /I "%1"==".DEFAULT_GOAL " GOTO .DEFAULT_GOAL 
IF /I "%1"=="for line in sys.stdin" GOTO for line in sys.stdin
IF /I "%1"=="BROWSER " GOTO BROWSER 
IF /I "%1"=="help" GOTO help
IF /I "%1"=="test" GOTO test
IF /I "%1"=="coverage" GOTO coverage
IF /I "%1"=="install" GOTO install
IF /I "%1"=="venv" GOTO venv
IF /I "%1"=="dev" GOTO dev
IF /I "%1"=="dev-venv" GOTO dev-venv
IF /I "%1"=="autoformat" GOTO autoformat
GOTO error

:.ONESHELL
	GOTO :EOF

:SHELL 
	CALL make.bat =
	CALL make.bat /bin/bash
	GOTO :EOF

:PACKAGE_NAME 
	CALL make.bat =
	CALL make.bat $(shell
	CALL make.bat basename
	CALL make.bat $(shell
	CALL make.bat dirname
	CALL make.bat $(realpath
	CALL make.bat $(lastword
	CALL make.bat $(MAKEFILE_LIST)))))
	GOTO :EOF

:.DEFAULT_GOAL 
	CALL make.bat =
	CALL make.bat help
	GOTO :EOF

:for line in sys.stdin
	match = re.match(r'^([a-zA-Z_-]+):.*?
	if match:
	target, help = match.groups()
	if not target.startswith('--'):
	print("%-20s %s" % (target, help))
	GOTO :EOF

:BROWSER 
	CALL make.bat =
	CALL make.bat python3
	CALL make.bat -c
	CALL make.bat "$$BROWSER_PYSCRIPT"
	GOTO :EOF

:help
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < %MAKEFILE_LIST%
	GOTO :EOF

:test
	pytest
	GOTO :EOF

:coverage
	pytest --cov-report term-missing --cov=%PACKAGE_NAME%
	GOTO :EOF

:install
	python3.7 -c "import %PACKAGE_NAME%" >/dev/null 2>&1 || python3 -m pip install . && python3.7 setup.py build_ext --inplace;
	GOTO :EOF

:venv
	@if ! command -v virtualenv >/dev/null 2>&1; then pip install virtualenv; fi && virtualenv ".%PACKAGE_NAME%_venv" -p python3 -q;
	GOTO :EOF

:dev
	CALL make.bat clean
	python3.7 -m pip install .[dev]
	GOTO :EOF

:dev-venv
	CALL make.bat venv
	GOTO :EOF

:autoformat
	black -l 120 %PACKAGE_NAME%
	GOTO :EOF

:error
    IF "%1"=="" (
        ECHO make: *** No targets specified and no makefile found.  Stop.
    ) ELSE (
        ECHO make: *** No rule to make target '%1%'. Stop.
    )
    GOTO :EOF
