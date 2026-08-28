@echo off
REM MFEM 示例 1:1 比对工具 (Windows cmd)
REM 用法: compare.bat ex1 ex2 ex3
REM       compare.bat --all

setlocal enabledelayedexpansion

set WIN_DIR=C:\Users\lilu\works\fem-pro\fem-rs
set DATA_DIR=%WIN_DIR%\data
set OUT_DIR=%WIN_DIR%\tmp\cmp
if not exist %OUT_DIR% mkdir %OUT_DIR%

set COUNT_OK=0
set COUNT_DIFF=0
set COUNT_FAIL=0
set COUNT_NOCPP=0
set COUNT_TOTAL=0

if "%1"=="--all" (
    for %%e in (ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9 ex10 ex11 ex12 ex13 ex14 ex15 ex16 ex17 ex18 ex19 ex20 ex21 ex22 ex23 ex24 ex25 ex26 ex27 ex28 ex29 ex30 ex31 ex32 ex33 ex34 ex35 ex36 ex37 ex38 ex39 ex40 ex41) do call :run_one %%e
    goto :summary
)

for %%e in (%*) do call :run_one %%e

:summary
echo.
echo ================================================================================
echo SUMMARY: OK=%COUNT_OK% DIFF=%COUNT_DIFF% FAIL=%COUNT_FAIL% NOCPP=%COUNT_NOCPP% TOTAL=%COUNT_TOTAL%
echo ================================================================================
goto :eof

:run_one
set /a COUNT_TOTAL+=1
set NAME=%1
set MESH=star.mesh
set RA=-no-vis
set CA=-no-vis
set MODE=dof

if "%NAME%"=="ex2" set MESH=beam-tri.mesh
if "%NAME%"=="ex3" set MESH=beam-tet.mesh
if "%NAME%"=="ex5" set MODE=dof+minres
if "%NAME%"=="ex6" set MESH=square-disc.mesh
if "%NAME%"=="ex7" set MESH=star.mesh
if "%NAME%"=="ex8" set MODE=dof+conv
if "%NAME%"=="ex9" set MESH=star.mesh
if "%NAME%"=="ex10" set MESH=beam-quad.mesh & set RA=-r 2 -o 2 -dt 3 -no-vis
if "%NAME%"=="ex11" set CA=-rs 0 & set MODE=eigenvalue
if "%NAME%"=="ex12" set MESH=beam-tri.mesh & set CA=-rs 0 -n 5 & set MODE=eigenvalue
if "%NAME%"=="ex13" set MESH=beam-tet.mesh & set RA=--ame -no-vis & set CA=-rs 0 -rp 0 & set MODE=eigenvalue
if "%NAME%"=="ex14" set CA=-r 4 -o 2 -no-vis
if "%NAME%"=="ex15" set MODE=dof+marked
if "%NAME%"=="ex16" set CA=-r 2 -o 2 -no-vis
if "%NAME%"=="ex17" set MESH=beam-tri.mesh
if "%NAME%"=="ex18" set MESH=periodic-square.mesh
if "%NAME%"=="ex19" set MESH=beam-quad.mesh & set RA=-o 2 -r 0 -no-vis
if "%NAME%"=="ex20" set MODE=dof+energy
if "%NAME%"=="ex21" set MESH=beam-tri.mesh & set RA=-o 2 -no-vis
if "%NAME%"=="ex22" set MESH=inline-quad.mesh & set RA=-p 0 -no-vis
if "%NAME%"=="ex23" set RA=-o 4 -tf 2 -no-vis
if "%NAME%"=="ex24" set RA=-p 2 -o 2 -no-vis
if "%NAME%"=="ex25" set MESH=inline-quad.mesh & set RA=-o 2 -f 5.0 -ref 3 -prob 4 -no-vis
if "%NAME%"=="ex26" set MESH=star.mesh
if "%NAME%"=="ex27" set MESH=inline-quad.mesh
if "%NAME%"=="ex28" set MESH=inline-quad.mesh
if "%NAME%"=="ex29" set MESH=disc-nurbs.mesh
if "%NAME%"=="ex30" set RA=-o 1 -no-vis & set MODE=dof+marked
if "%NAME%"=="ex31" set MESH=beam-tri.mesh & set RA=-o 1 -r 1 -no-vis
if "%NAME%"=="ex32" set MESH=fichera.mesh & set CA=-rs 0 & set MODE=eigenvalue
if "%NAME%"=="ex33" set MESH=square-disc.mesh & set RA=-alpha 0.33 -o 2 -no-vis
if "%NAME%"=="ex34" set MESH=fichera-mixed.mesh
if "%NAME%"=="ex35" set MESH=fichera-mixed.mesh & set RA=-p 0 -o 1 -no-vis & set CA=-p 0 -o 1 -no-vis -rs 0
if "%NAME%"=="ex36" set MESH=disc-nurbs.mesh
if "%NAME%"=="ex37" set MODE=dof+objective
if "%NAME%"=="ex38" set MESH=inline-segment.mesh
if "%NAME%"=="ex39" set MESH=compass.msh
if "%NAME%"=="ex40" set MESH=star.mesh
if "%NAME%"=="ex41" set MESH=periodic-square.mesh & set RA=-p 0 -r 2 -o 3 -no-vis

echo === %NAME% ===

REM --- Rust ---
"%WIN_DIR%\target\release\examples\%NAME%.exe" -m "%DATA_DIR%\%MESH%" %RA% > "%OUT_DIR%\%NAME%_rust.log" 2>&1
if errorlevel 1 (
    set /a COUNT_FAIL+=1
    echo   FAIL
    goto :eof
)

REM --- C++ ---
REM Try serial binary first, then parallel
if exist "\\wsl$\Ubuntu\home\quan\bin\%NAME%_cpp" (
    wsl -e bash -c "timeout 150 ~/bin/%NAME%_cpp -m /home/quan/mfem49/data/%MESH% %CA%" > "%OUT_DIR%\%NAME%_cpp.log" 2>&1
) else (
    wsl -e bash -c "timeout 150 ~/bin/%NAME%p_cpp -m /home/quan/mfem49/data/%MESH% -rs 0 %CA%" > "%OUT_DIR%\%NAME%_cpp.log" 2>&1
)

REM --- Compare ---
set RUST_VAL=
set CPP_VAL=
for /f "usebackq tokens=*" %%a in (`python -c "import re; t=open(r'%OUT_DIR%\%NAME%_rust.log').read(); m=re.search(r'Number of (finite element )?unknowns: (\d+)', t); print(m.group(2) if m else '')"`) do set RUST_VAL=%%a
for /f "usebackq tokens=*" %%a in (`python -c "import re; t=open(r'%OUT_DIR%\%NAME%_cpp.log').read(); m=re.search(r'Number of (finite element )?unknowns: (\d+)', t); print(m.group(2) if m else '')"`) do set CPP_VAL=%%a

if "%RUST_VAL%"=="" (
    set /a COUNT_FAIL+=1
    echo   NO_RUST_DOF
) else if "%CPP_VAL%"=="" (
    set /a COUNT_NOCPP+=1
    echo   NO_CPP_DOF
) else if "%RUST_VAL%"=="%CPP_VAL%" (
    set /a COUNT_OK+=1
    echo   OK (dof=%RUST_VAL%)
) else (
    set /a COUNT_DIFF+=1
    echo   DIFF (rust=%RUST_VAL% cpp=%CPP_VAL%)
)
goto :eof
