# Registers a Task Scheduler task that starts the voice agent at login, with no console window.
# Run once:  powershell -ExecutionPolicy Bypass -File install_autostart.ps1
# Remove:    powershell -ExecutionPolicy Bypass -File install_autostart.ps1 -Remove
param([switch]$Remove)
$name = "VoiceAgent"
if ($Remove) { Unregister-ScheduledTask -TaskName $name -Confirm:$false; "removed"; exit }

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyw  = Join-Path $here ".venv\Scripts\pythonw.exe"
$sup  = Join-Path $here "run_hidden.pyw"
$user = "$env:USERDOMAIN\$env:USERNAME"

$action    = New-ScheduledTaskAction -Execute $pyw -Argument "`"$sup`"" -WorkingDirectory $here
$trigger   = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
"Installed task '$name'. It starts at your next login. Start it now with: Start-ScheduledTask -TaskName $name"
