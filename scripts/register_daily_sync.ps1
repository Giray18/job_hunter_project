<#
Registers a Windows scheduled task that runs the morning sync every day:
  waits for today's Databricks run of job_hunter_ingestion, downloads the applications to
  %USERPROFILE%\Downloads\JobApplications\<date>\ and opens index.html.

Run once from the repo root in PowerShell:
  powershell -ExecutionPolicy Bypass -File scripts\register_daily_sync.ps1 [-At 09:20] [-Profile personal-projects]

The task runs only while you are logged on (it opens the page in your browser). If the PC was off at the
scheduled time, it starts as soon as possible afterwards. Remove it with:
  Unregister-ScheduledTask -TaskName "JobHunter daily sync" -Confirm:$false
#>
param(
    [string]$At = "09:20",
    [string]$Profile = "personal-projects",
    [string]$TaskName = "JobHunter daily sync"
)

$repo = Split-Path -Parent $PSScriptRoot
$uv = (Get-Command uv -ErrorAction Stop).Source

$action = New-ScheduledTaskAction -Execute $uv `
    -Argument "run python -m job_hunter.sync_today --profile $Profile --open" `
    -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "Downloads today's job applications from Databricks (job_hunter)" -Force |
    Out-Null
Write-Output "Registered '$TaskName': daily at $At, runs '$uv run python -m job_hunter.sync_today' in $repo"
