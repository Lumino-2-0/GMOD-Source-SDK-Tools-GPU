# Regenere flow_gpu_kernel_ptx.h depuis flow_gpu_kernel.cu (necessite nvcc + VS).
# A relancer uniquement quand flow_gpu_kernel.cu change. Le PTX est portable :
# le pilote NVIDIA le compile (JIT) pour tout GPU >= compute 7.5 (Turing+) ;
# les GPU NVIDIA plus anciens retombent sur OpenCL.
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$cu  = Join-Path $here 'flow_gpu_kernel.cu'
$ptx = Join-Path $env:TEMP 'flow_gpu_kernel.ptx'
$out = Join-Path $here 'flow_gpu_kernel_ptx.h'

# nvcc exige le compilateur hote x64 (le cl.exe du PATH peut etre x86).
$vcvars = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio' -Recurse -Filter vcvars64.bat -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $vcvars) { throw 'vcvars64.bat introuvable' }

cmd /c "`"$($vcvars.FullName)`" >nul && nvcc -ptx -arch=compute_75 -fmad=false -prec-div=true -prec-sqrt=true -o `"$ptx`" `"$cu`""
if ($LASTEXITCODE -ne 0) { throw 'nvcc a echoue' }

$bytes = [System.IO.File]::ReadAllBytes($ptx)
$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine('// Genere par build_ptx.ps1 depuis flow_gpu_kernel.cu - NE PAS EDITER.')
[void]$sb.AppendLine('#pragma once')
[void]$sb.AppendLine('static const unsigned char g_RaycastPTX[] = {')
for ($i = 0; $i -lt $bytes.Length; $i += 24) {
	$n = [Math]::Min(24, $bytes.Length - $i)
	$line = ($bytes[$i..($i + $n - 1)] | ForEach-Object { $_.ToString() }) -join ','
	[void]$sb.AppendLine($line + ',')
}
[void]$sb.AppendLine('0 };')
[System.IO.File]::WriteAllText($out, $sb.ToString())
Write-Host "OK : $($bytes.Length) octets de PTX -> $out"
