@echo off
setlocal enabledelayedexpansion

cd /d E:\Research\RAG

echo === Git Repository Initialization ===
echo.

REM Check if git repo exists
if exist .git (
    echo [OK] Git repository already initialized
) else (
    echo [INIT] Initializing git repository...
    git init
    if errorlevel 1 (
        echo [ERROR] Failed to initialize git repository
        exit /b 1
    )
    echo [OK] Git repository initialized
)

echo.
echo === Checking current branch ===
for /f "tokens=*" %%i in ('git branch --show-current') do set CURRENT_BRANCH=%%i
echo Current branch: !CURRENT_BRANCH!

if "!CURRENT_BRANCH!"=="" (
    echo [WARN] No commits yet, branch name not set
    set CURRENT_BRANCH=master
)

if not "!CURRENT_BRANCH!"=="main" (
    echo [RENAME] Renaming branch from !CURRENT_BRANCH! to main...
    git branch -M main
    if errorlevel 1 (
        echo [ERROR] Failed to rename branch
        exit /b 1
    )
)

echo [OK] Default branch is now: main

echo.
echo === Configuring remote origin ===
for /f "tokens=*" %%i in ('git config --get remote.origin.url 2^>nul') do set EXISTING_URL=%%i

if defined EXISTING_URL (
    echo Existing remote URL: !EXISTING_URL!
    if not "!EXISTING_URL!"=="git@github.com:Wesirehour/AI-Reader.git" (
        echo [UPDATE] Updating remote origin URL...
        git remote set-url origin git@github.com:Wesirehour/AI-Reader.git
    )
) else (
    echo [ADD] Adding remote origin...
    git remote add origin git@github.com:Wesirehour/AI-Reader.git
)

echo [OK] Remote origin: git@github.com:Wesirehour/AI-Reader.git

echo.
echo === Staging and committing files ===
git add -A
echo [OK] Files staged

echo [COMMIT] Creating commit...
git commit -m "chore: initialize AI-Reader project" -m "" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
if errorlevel 1 (
    echo [WARN] Commit failed - checking if already committed
    for /f "tokens=*" %%i in ('git rev-parse HEAD 2^>nul') do set COMMIT_HASH=%%i
    if defined COMMIT_HASH (
        echo [OK] Repository already has commits
    ) else (
        echo [ERROR] Failed to create commit
        exit /b 1
    )
) else (
    echo [OK] Commit created successfully
)

echo.
echo === Getting commit information ===
for /f "tokens=*" %%i in ('git rev-parse HEAD') do set COMMIT_HASH=%%i
echo Latest commit hash: !COMMIT_HASH!

echo.
echo === Pushing to GitHub ===
git push -u origin main
if errorlevel 1 (
    echo [ERROR] Push failed
    git push -u origin main 2>&1
    exit /b 1
)

echo [OK] Push successful

echo.
echo === Final Status ===
echo Current branch: main
echo Remote URL: git@github.com:Wesirehour/AI-Reader.git
echo Latest commit: !COMMIT_HASH!
echo [SUCCESS] Repository initialized and pushed to GitHub

exit /b 0
