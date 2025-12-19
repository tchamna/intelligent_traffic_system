# Setup script: create virtual environment and install requirements
$ErrorActionPreference = 'Stop'

$venv = '.venv'

function Find-PythonInvoker {
    if (Get-Command python -ErrorAction SilentlyContinue) { return 'python' }
    if (Get-Command py -ErrorAction SilentlyContinue) { return 'py -3' }
    return $null
}

$invoker = Find-PythonInvoker
if (-not $invoker) {
    Write-Error "No Python launcher found. Install Python 3.10+ and ensure 'python' or 'py' is available on PATH."
    exit 1
}

if (-Not (Test-Path $venv)) {
    Write-Host "Creating virtual environment in $venv..."
    & $invoker -m venv $venv
} else {
    Write-Host "Virtual environment $venv already exists."
}

$venvPython = Join-Path $venv 'Scripts\python.exe'
$venvPip = Join-Path $venv 'Scripts\pip.exe'

if (-Not (Test-Path $venvPython)) {
    Write-Error "Virtual environment Python not found at $venvPython"
    exit 1
}

Write-Host "Upgrading pip in virtual environment..."
& $venvPython -m pip install --upgrade pip

Write-Host "Installing requirements..."
& $venvPip install -r .\requirements.txt

Write-Host "Setup complete. To activate the venv later, run: .\$venv\Scripts\Activate.ps1"
