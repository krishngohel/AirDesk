$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
if (-not (Test-Path "$root\.venv")) {
    py -3.11 -m venv "$root\.venv"
    & "$root\.venv\Scripts\python.exe" -m pip install --upgrade pip
    & "$root\.venv\Scripts\pip.exe" install -r "$root\requirements.txt"
    & "$root\.venv\Scripts\pip.exe" install -r "$root\requirements-voice.txt"
}
& "$root\.venv\Scripts\python.exe" -m airdesk.main @args
