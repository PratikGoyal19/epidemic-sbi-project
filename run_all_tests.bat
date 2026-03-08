@echo off
echo ======================================================================
echo RUNNING ALL TESTS
echo ======================================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate

echo ======================================================================
echo TEST 1: SIMULATOR
echo ======================================================================
python 01_simulator\sir_model.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ TEST 1 FAILED!
    pause
    exit /b 1
)
echo.
echo ✅ TEST 1 PASSED!
echo.
pause

echo ======================================================================
echo TEST 2: DATA GENERATION
echo ======================================================================
cd 02_data
python test_generation.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ TEST 2 FAILED!
    cd ..
    pause
    exit /b 1
)
cd ..
echo.
echo ✅ TEST 2 PASSED!
echo.
pause

echo ======================================================================
echo TEST 3: NPE TRAINING
echo ======================================================================
cd 03_methods
python test_npe.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ TEST 3 FAILED!
    cd ..
    pause
    exit /b 1
)
cd ..
echo.
echo ✅ TEST 3 PASSED!
echo.

echo ======================================================================
echo 🎉 ALL TESTS PASSED!
echo ======================================================================
echo.
echo ✅ Simulator works
echo ✅ Data generation works
echo ✅ NPE training works
echo.
echo 🚀 Ready for full data generation and training!
echo.
pause