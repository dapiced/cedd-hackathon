#====================================================================================================================================================
# Open a PowerShell window and run these 2 commands:
#
# Unblock-File -Path .\setup-cedd-win.ps1
# .\setup-cedd-win.ps1
# 
# PS: The installer will probably ask you to run the PowerShell script three times (1st: install Python, 2nd: install Git, 3rd: pull the CEDD repo).
# ====================================================================================================================================================
# CEDD — Setup Script for Windows 11
# Hackathon Mila x Bell x Kids Help Phone (March 16-23, 2026)
# Team: 404HarmNotFound
# ====================================================================================================================================================
# This script installs everything needed to run CEDD on Windows 11:
#   - Checks/installs Python 3.11
#   - Clones the GitHub repo (or git pull if already cloned)
#   - Creates a virtual environment
#   - Installs pip dependencies
#   - Trains the model
#   - Launches the Streamlit application
#
# Re-run: if the repo is already cloned, the script will offer to
# run "git pull origin main" to fetch the latest changes.
#
# Usage:
#   FIRST TIME — if PowerShell blocks script execution, run ONE of these:
#     Option A (permanent fix, recommended):
#       Unblock-File -Path .\setup-cedd.ps1
#       .\setup-cedd.ps1
#
#     Option B (one-time bypass, no permanent change):
#       Unblock-File -Path .\setup-cedd.ps1
#
#   NORMAL USAGE:
#     .\setup-cedd.ps1
#     .\setup-cedd.ps1 -AnthropicApiKey "sk-ant-..."
#     .\setup-cedd.ps1 -SkipClone   (if repo is already cloned)
#=========================================================================================================================================================
param(
    [string]$AnthropicApiKey = "",
    [string]$InstallDir = "$HOME\cedd-hackathon",
    [switch]$SkipClone,
    [switch]$SkipTrain,
    [switch]$LaunchApp
)

# --- Colors and utility functions ---
function Write-Step  { param([string]$msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Ok    { param([string]$msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn  { param([string]$msg) Write-Host "[!]  $msg" -ForegroundColor Yellow }
function Write-Err   { param([string]$msg) Write-Host "[X]  $msg" -ForegroundColor Red }

$ErrorActionPreference = "Stop"

# --- Check ExecutionPolicy (protects venv activation later) ---
$currentPolicy = Get-ExecutionPolicy -Scope CurrentUser
if ($currentPolicy -eq "Restricted" -or $currentPolicy -eq "Undefined") {
    $systemPolicy = Get-ExecutionPolicy -Scope LocalMachine
    if ($systemPolicy -eq "Restricted" -or $systemPolicy -eq "Undefined") {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host "  WARNING: ExecutionPolicy is '$currentPolicy'" -ForegroundColor Red
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host ""
        Write-Host "  Other scripts (like venv activation) may be blocked." -ForegroundColor Yellow
        Write-Host "  This is a one-time fix that only affects YOUR user account." -ForegroundColor Yellow
        Write-Host ""
        $choice = Read-Host "  Set ExecutionPolicy to RemoteSigned for your user? (y/n)"
        if ($choice -eq "y" -or $choice -eq "Y") {
            try {
                Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
                Write-Host "[OK] ExecutionPolicy set to RemoteSigned." -ForegroundColor Green
                Write-Host ""
            } catch {
                Write-Host "[!]  Could not set ExecutionPolicy automatically." -ForegroundColor Yellow
                Write-Host "     Run this command manually:" -ForegroundColor Yellow
                Write-Host "     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned" -ForegroundColor White
                Write-Host "     Then re-run the script." -ForegroundColor Yellow
                exit 1
            }
        } else {
            Write-Host ""
            Write-Host "[!]  Continuing, but venv activation may fail later." -ForegroundColor Yellow
            Write-Host "     If it does, run: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned" -ForegroundColor Yellow
            Write-Host ""
        }
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  CEDD - Conversational Emotional Drift Detection"           -ForegroundColor Magenta
Write-Host "  Windows 11 Setup - Team 404HarmNotFound"                   -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
Write-Host "  IMPORTANT — LLM Options for Conversation"                  -ForegroundColor Yellow
Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
Write-Host ""
Write-Host "  CEDD has two layers:" -ForegroundColor White
Write-Host "    1. DETECTION (features, classifier, alerts, charts)" -ForegroundColor White
Write-Host "       -> ALWAYS works, no prerequisites." -ForegroundColor Green
Write-Host ""
Write-Host "    2. CONVERSATION (intelligent chatbot responses)" -ForegroundColor White
Write-Host "       -> Requires an LLM for contextual responses." -ForegroundColor White
Write-Host ""
Write-Host "  Without an LLM, the chatbot replies with static" -ForegroundColor Yellow
Write-Host "  pre-defined phrases per alert level (no context tracking," -ForegroundColor Yellow
Write-Host "  no real interaction). Detection works fine, but the" -ForegroundColor Yellow
Write-Host "  conversation experience is very limited." -ForegroundColor Yellow
Write-Host ""
Write-Host "  3 options to enable the LLM:" -ForegroundColor Cyan
Write-Host "    Option 1 (recommended): Anthropic API Key (Claude)" -ForegroundColor White
Write-Host "      -> Ask Dominic for the key or create an account at" -ForegroundColor Gray
Write-Host "         https://console.anthropic.com (free credits available)" -ForegroundColor Gray
Write-Host "      -> Run: .\setup-cedd.ps1 -AnthropicApiKey 'sk-ant-...'" -ForegroundColor Gray
Write-Host ""
Write-Host "    Option 2: Local Ollama (free, ~4 GB RAM required)" -ForegroundColor White
Write-Host "      -> Install Ollama: https://ollama.com/download" -ForegroundColor Gray
Write-Host "      -> Then run: ollama pull mistral" -ForegroundColor Gray
Write-Host ""
Write-Host "    Option 3: No LLM (static responses only)" -ForegroundColor White
Write-Host "      -> Run: .\setup-cedd.ps1 (without -AnthropicApiKey)" -ForegroundColor Gray
Write-Host ""
Write-Host "  Automatic fallback: Claude -> Mistral -> Llama -> No LLM" -ForegroundColor Cyan
Write-Host "  The system automatically uses the best available LLM." -ForegroundColor Cyan
Write-Host ""
Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to continue with the installation"

# ============================================================================
# STEP 1: Check Python
# ============================================================================
Write-Step "Step 1/6: Checking Python"

$pythonCmd = $null

# Look for python in order: python, python3, py
foreach ($candidate in @("python", "python3", "py")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 9) {
                $pythonCmd = $candidate
                Write-Ok "Python found: $ver (command: $candidate)"
                break
            } else {
                Write-Warn "$candidate = $ver (minimum required: 3.9+)"
            }
        }
    } catch {
        # This candidate doesn't exist, move on
    }
}

if (-not $pythonCmd) {
    Write-Err "Python 3.9+ not found on this system."
    Write-Host ""
    Write-Host "Installation options:" -ForegroundColor Yellow
    Write-Host "  1. Microsoft Store: search for 'Python 3.11' in the Store"
    Write-Host "  2. Official site  : https://www.python.org/downloads/"
    Write-Host ""
    Write-Host "IMPORTANT during installation:" -ForegroundColor Yellow
    Write-Host "  - Check 'Add Python to PATH' (at the bottom of the installer)"
    Write-Host "  - Check 'Install pip'"
    Write-Host ""

    $choice = Read-Host "Would you like the script to try installing Python 3.11 via winget? (y/n)"
    if ($choice -eq "y" -or $choice -eq "Y") {
        Write-Step "Installing Python 3.11 via winget..."
        try {
            winget install Python.Python.3.11 --accept-package-agreements --accept-source-agreements
            Write-Ok "Python 3.11 installed via winget."
            Write-Warn "You must CLOSE and REOPEN PowerShell for Python to be in the PATH."
            Write-Warn "Then re-run this script: .\setup-cedd.ps1"
            exit 0
        } catch {
            Write-Err "winget failed. Install Python manually from https://www.python.org/downloads/"
            exit 1
        }
    } else {
        Write-Host "Install Python 3.9+ then re-run this script." -ForegroundColor Yellow
        exit 1
    }
}

# Check that pip is available
Write-Step "Checking pip"
try {
    $pipVer = & $pythonCmd -m pip --version 2>&1
    Write-Ok "pip found: $pipVer"
} catch {
    Write-Warn "pip not found. Attempting installation..."
    & $pythonCmd -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Unable to install pip. Check your Python installation."
        exit 1
    }
    Write-Ok "pip installed."
}

# Check that venv is available
Write-Step "Checking venv module"
try {
    & $pythonCmd -c "import venv" 2>&1 | Out-Null
    Write-Ok "venv module available."
} catch {
    Write-Err "The venv module is not available."
    Write-Host "If you installed Python from the Microsoft Store, venv should be included." -ForegroundColor Yellow
    Write-Host "Otherwise, reinstall Python and make sure 'pip' and 'venv' are checked." -ForegroundColor Yellow
    exit 1
}

# ============================================================================
# STEP 2: Check Git and clone/update the repo
# ============================================================================
Write-Step "Step 2/6: GitHub Repo (clone or update)"

if (-not $SkipClone) {
    # Check git
    try {
        $gitVer = git --version 2>&1
        Write-Ok "Git found: $gitVer"
    } catch {
        Write-Err "Git not found."
        Write-Host "Options:" -ForegroundColor Yellow
        Write-Host "  winget install Git.Git"
        Write-Host "  or: https://git-scm.com/download/win"
        
        $choice = Read-Host "Try installing Git via winget? (y/n)"
        if ($choice -eq "y" -or $choice -eq "Y") {
            winget install Git.Git --accept-package-agreements --accept-source-agreements
            Write-Warn "Git installed. Close and reopen PowerShell, then re-run this script."
            exit 0
        } else {
            exit 1
        }
    }

    if (Test-Path (Join-Path $InstallDir ".git")) {
        # Repo already exists — offer to update
        Write-Ok "Existing repo detected in $InstallDir"
        $choice = Read-Host "Update the repo with the latest changes (git pull)? (y/n)"
        if ($choice -eq "y" -or $choice -eq "Y") {
            Push-Location $InstallDir
            try {
                # Check for uncommitted local changes
                $status = git status --porcelain 2>&1
                if ($status) {
                    Write-Warn "Uncommitted local changes detected:"
                    Write-Host $status -ForegroundColor Yellow
                    Write-Warn "git pull will attempt a merge. Resolve conflicts manually if needed."
                }

                Write-Host "git pull origin main..." -ForegroundColor Gray
                git pull origin main 2>&1
                if ($LASTEXITCODE -ne 0) {
                    Write-Err "git pull failed. Check your connection or resolve conflicts."
                    Pop-Location
                    exit 1
                }
                Write-Ok "Repo updated (git pull)."
            } finally {
                Pop-Location
            }
        } else {
            Write-Ok "Update skipped. Keeping local version."
        }
    } elseif (Test-Path $InstallDir) {
        # Folder exists but is not a git repo
        Write-Warn "Folder $InstallDir exists but is not a git repo."
        $choice = Read-Host "Delete and clone the repo? (y/n)"
        if ($choice -eq "y" -or $choice -eq "Y") {
            Remove-Item -Recurse -Force $InstallDir
        } else {
            Write-Err "Cannot continue without a valid git repo."
            exit 1
        }
    }

    if (-not (Test-Path (Join-Path $InstallDir ".git"))) {
        git clone https://github.com/dapiced/cedd-hackathon.git $InstallDir
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Clone failed. Check your internet connection."
            exit 1
        }
        Write-Ok "Repo cloned to $InstallDir"
    }
} else {
    Write-Ok "Clone skipped (-SkipClone)."
}

Set-Location $InstallDir
Write-Ok "Working directory: $(Get-Location)"

# ============================================================================
# STEP 3: Create the virtual environment
# ============================================================================
Write-Step "Step 3/6: Creating virtual environment"

$venvDir = Join-Path $InstallDir "venv"
$activateScript = Join-Path $venvDir "Scripts\Activate.ps1"

if (Test-Path $venvDir) {
    Write-Warn "venv already exists in $venvDir"
    $choice = Read-Host "Recreate the venv? (y/n)"
    if ($choice -eq "y" -or $choice -eq "Y") {
        Remove-Item -Recurse -Force $venvDir
        & $pythonCmd -m venv $venvDir
        Write-Ok "venv recreated."
    } else {
        Write-Ok "Keeping existing venv."
    }
} else {
    & $pythonCmd -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to create venv."
        exit 1
    }
    Write-Ok "venv created in $venvDir"
}

# Activate the venv
Write-Step "Activating virtual environment"
& $activateScript
Write-Ok "venv activated."

# ============================================================================
# STEP 4: Install dependencies
# ============================================================================
Write-Step "Step 4/6: Installing Python dependencies"

# Upgrade pip first
Write-Host "Upgrading pip..." -ForegroundColor Gray
python -m pip install --upgrade pip 2>&1 | Out-Null

# Check if requirements.txt exists
$reqFile = Join-Path $InstallDir "requirements.txt"
if (Test-Path $reqFile) {
    Write-Host "Installing from requirements.txt..." -ForegroundColor Gray
    python -m pip install -r $reqFile
} else {
    Write-Warn "requirements.txt not found. Installing dependencies manually..."
    $packages = @(
        "streamlit",
        "plotly",
        "scikit-learn",
        "numpy",
        "joblib",
        "requests",
        "anthropic"
    )
    foreach ($pkg in $packages) {
        Write-Host "  Installing $pkg..." -ForegroundColor Gray
        python -m pip install $pkg
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Failed to install $pkg"
            exit 1
        }
    }
}

Write-Ok "All dependencies installed."

# Show package summary
Write-Host "`nInstalled packages:" -ForegroundColor Gray
python -m pip list --format=columns | Select-String -Pattern "streamlit|plotly|scikit|numpy|joblib|requests|anthropic"

# ============================================================================
# STEP 5: Configure the Anthropic API key (optional)
# ============================================================================
Write-Step "Step 5/6: Anthropic API Key Configuration"

if ($AnthropicApiKey -ne "") {
    # Set for the current session
    $env:ANTHROPIC_API_KEY = $AnthropicApiKey
    Write-Ok "API key set for this PowerShell session."
    
    # Offer to save it permanently
    $choice = Read-Host "Save the key as a permanent user environment variable? (y/n)"
    if ($choice -eq "y" -or $choice -eq "Y") {
        [System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", $AnthropicApiKey, "User")
        Write-Ok "API key saved as a user environment variable."
        Write-Warn "It will be available in all new PowerShell windows."
    }
} else {
    # Check if it already exists
    $existingKey = [System.Environment]::GetEnvironmentVariable("ANTHROPIC_API_KEY", "User")
    if ($existingKey) {
        $env:ANTHROPIC_API_KEY = $existingKey
        Write-Ok "API key found in user environment variables."
    } else {
        Write-Warn "No Anthropic API key configured."
        Write-Host "  The app will run in 'no LLM' mode (pre-defined responses)." -ForegroundColor Yellow
        Write-Host "  To enable Claude, re-run with: .\setup-cedd.ps1 -AnthropicApiKey 'sk-ant-...'" -ForegroundColor Yellow
        Write-Host "  Or set manually: `$env:ANTHROPIC_API_KEY = 'sk-ant-...'" -ForegroundColor Yellow
    }
}

# ============================================================================
# STEP 6: Train the model and (optionally) launch the app
# ============================================================================
Write-Step "Step 6/6: Training the model"

if (-not $SkipTrain) {
    Write-Host "Training the CEDD classifier..." -ForegroundColor Gray
    python train.py
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Training failed. Check the files in data/."
        exit 1
    }
    Write-Ok "Model trained. File: models/cedd_model.joblib"
} else {
    Write-Ok "Training skipped (-SkipTrain)."
}

# ============================================================================
# FINAL SUMMARY
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Directory : $InstallDir" -ForegroundColor White
Write-Host "  Python    : $pythonCmd" -ForegroundColor White

if ($env:ANTHROPIC_API_KEY) {
    Write-Host "  API Key   : configured (Claude active)" -ForegroundColor White
} else {
    Write-Host "  API Key   : not configured (no LLM mode)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  To launch the application:" -ForegroundColor Cyan
Write-Host "    cd $InstallDir" -ForegroundColor White
Write-Host "    .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "    streamlit run app.py" -ForegroundColor White
Write-Host ""
Write-Host "  To generate more synthetic data:" -ForegroundColor Cyan
Write-Host "    python generate_synthetic_data.py --lang fr --count 20" -ForegroundColor White
Write-Host "    python generate_synthetic_data.py --lang en --count 20" -ForegroundColor White
Write-Host ""
Write-Host "  To simulate session history (demo):" -ForegroundColor Cyan
Write-Host "    python simulate_history.py --lang fr" -ForegroundColor White
Write-Host ""

if ($LaunchApp) {
    Write-Step "Launching the application..."
    Write-Host "The interface will open in your browser at http://localhost:8501" -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Yellow
    streamlit run app.py
}
