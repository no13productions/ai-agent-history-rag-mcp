# Install ai-agent-history-rag daemon as a Windows scheduled task (runs at login)
#
# Run this script in PowerShell as your normal user (not admin):
#   .\scripts\install-windows.ps1
#
# To configure environment variables, edit this script before running,
# or set them in your user environment variables.

$ErrorActionPreference = "Stop"

$TaskName = "AIAgentHistoryRAG"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

# Find uv.exe
$UvPath = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $UvPath) {
    $UvPath = "$env:USERPROFILE\.local\bin\uv.exe"
}
if (-not (Test-Path $UvPath)) {
    $UvPath = "$env:LOCALAPPDATA\uv\uv.exe"
}
if (-not (Test-Path $UvPath)) {
    Write-Error "uv not found. Install it first: irm https://astral.sh/uv/install.ps1 | iex"
    exit 1
}

Write-Host "Using uv at: $UvPath"
Write-Host "Project directory: $ProjectDir"

# Create data directory
$DataDir = "$env:USERPROFILE\.claude-history-rag"
if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir | Out-Null
}

# Remove existing task if present
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Build the command arguments
$Arguments = "--directory `"$ProjectDir`" run ai-agent-history-rag-daemon start"

# Create the scheduled task action
$Action = New-ScheduledTaskAction -Execute $UvPath -Argument $Arguments -WorkingDirectory $ProjectDir

# Create trigger to run at logon
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Create settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

# Register the task
Write-Host "Creating scheduled task..."
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "AI Agent History RAG Daemon - indexes AI coding agent history"

# Start the task now
Write-Host "Starting daemon..."
Start-ScheduledTask -TaskName $TaskName

Write-Host ""
Write-Host "Done! Daemon installed and started." -ForegroundColor Green
Write-Host ""
Write-Host "Commands:"
Write-Host "  Status:    Get-ScheduledTask -TaskName $TaskName"
Write-Host "  Stop:      Stop-ScheduledTask -TaskName $TaskName"
Write-Host "  Start:     Start-ScheduledTask -TaskName $TaskName"
Write-Host "  Uninstall: Unregister-ScheduledTask -TaskName $TaskName"
Write-Host ""
Write-Host "Logs: $DataDir\daemon.log"
Write-Host ""
Write-Host "To set environment variables (e.g., for client mode):"
Write-Host "  1. Open System Properties > Environment Variables"
Write-Host "  2. Add user variables like CLAUDE_HISTORY_RAG_SERVER_URL"
Write-Host "  3. Restart the task: Stop-ScheduledTask $TaskName; Start-ScheduledTask $TaskName"
Write-Host ""
