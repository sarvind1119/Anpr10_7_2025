$python = if (Test-Path "$PSScriptRoot\.venv\Scripts\python.exe") {
  "$PSScriptRoot\.venv\Scripts\python.exe"
} else {
  "$PSScriptRoot\venv\Scripts\python.exe"
}
& $python -m app.cli dev
