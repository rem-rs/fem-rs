@echo off
REM Run a Rust example and capture output
REM Usage: run_rust.bat <exe_path> <mesh_path> <args> <output_file>

set EXE=%~1
set MESH=%~2
set ARGS=%~3
set OUT=%~4

"%EXE%" -m "%MESH%" %ARGS% > "%OUT%" 2>&1
echo EXIT=%ERRORLEVEL% >> "%OUT%"
