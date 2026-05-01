<#
.SYNOPSIS
    Wire a Curry MCP server into the Claude desktop app.

.DESCRIPTION
    - Finds a Python interpreter that has the mcp package installed.
    - Initialises a project directory (.curry/config.json + core DB model
      registration) if it does not already exist.
    - Merges a new mcpServers entry into claude_desktop_config.json without
      disturbing existing entries.
    - Prints restart instructions when done.

.PARAMETER ProjectDir
    Path to the project directory to wire.  A .curry/config.json will be
    created there if one does not exist.  Defaults to an interactive prompt.

.PARAMETER ServerName
    Key used in claude_desktop_config.json mcpServers.  Defaults to
    "curry-<projectname>".

.PARAMETER Force
    Overwrite an existing mcpServers entry with the same name.

.EXAMPLE
    .\Add-CurryToClaudeDesktop.ps1 -ProjectDir C:\Projects\my-ai-pipeline

.EXAMPLE
    .\Add-CurryToClaudeDesktop.ps1 -ProjectDir C:\Projects\my-ai-pipeline -ServerName curry-pipeline -Force
#>

[CmdletBinding()]
param(
    [string]$ProjectDir  = "",
    [string]$ServerName  = "",
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ─── Paths ────────────────────────────────────────────────────────────────────

$CURRY_ROOT   = "C:\AI-Local\Curry"
$MCP_SERVER   = Join-Path $CURRY_ROOT "curry_mcp_server.py"
$CORE_DB      = Join-Path $CURRY_ROOT "curry_core.db"
$CLAUDE_CFG   = "$env:APPDATA\Claude\claude_desktop_config.json"

# ─── Helpers ──────────────────────────────────────────────────────────────────

function Write-Step  { param($msg) Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Write-OK    { param($msg) Write-Host "   [OK] $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "   [!!] $msg" -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "`n[FAIL] $msg" -ForegroundColor Red ; exit 1 }

# ─── 1. Locate a Python with mcp installed ────────────────────────────────────

Write-Step "Locating Python with mcp package"

# Candidates in preference order: venv used by Curry, system Python 3.12, PATH
$pythonCandidates = @(
    (Join-Path $CURRY_ROOT ".venv\Scripts\python.exe"),
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python312\python.exe",
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python311\python.exe",
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python310\python.exe",
    (Get-Command python  -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
    (Get-Command python3 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
) | Where-Object { $_ -and (Test-Path $_ -ErrorAction SilentlyContinue) } | Select-Object -Unique

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
    $check = & $candidate -c "import mcp, sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $check) {
        $pythonExe = $check.Trim()
        Write-OK "Found: $pythonExe"
        break
    }
}

if (-not $pythonExe) {
    Write-Warn "No Python with mcp found in standard locations."
    Write-Warn "Install it with:  pip install mcp"
    Write-Warn "Then re-run this script."
    Write-Fail "Cannot continue without mcp."
}

# ─── 2. Validate Curry installation ───────────────────────────────────────────

Write-Step "Validating Curry installation"

if (-not (Test-Path $MCP_SERVER)) {
    Write-Fail "curry_mcp_server.py not found at $MCP_SERVER`nIs CURRY_ROOT set correctly?"
}
Write-OK "MCP server: $MCP_SERVER"

if (-not (Test-Path $CORE_DB)) {
    Write-Warn "curry_core.db not found.  It will be created when the server first starts."
    Write-Warn "Register at least one model with Curry before running inferences:"
    Write-Warn "  python -c `"from curry_core import Curry; db=Curry('$CORE_DB'); db.register_model('claude-sonnet-4-6',1,'release_20260101',temperature=1.0,top_p=0.999,max_tokens=8192)`""
} else {
    Write-OK "Core DB: $CORE_DB"
}

# ─── 3. Resolve project directory ─────────────────────────────────────────────

Write-Step "Resolving project directory"

if (-not $ProjectDir) {
    $ProjectDir = Read-Host "   Enter project directory path (will be created if missing)"
}
$ProjectDir = $ProjectDir.Trim().TrimEnd('\')

if (-not $ProjectDir) {
    Write-Fail "No project directory supplied."
}

# Create directory tree if needed
$curryDir = Join-Path $ProjectDir ".curry"
if (-not (Test-Path $curryDir)) {
    New-Item -ItemType Directory -Path $curryDir -Force | Out-Null
    Write-OK "Created: $curryDir"
} else {
    Write-OK "Exists:  $curryDir"
}

# ─── 4. Write .curry/config.json if missing ───────────────────────────────────

Write-Step "Checking .curry/config.json"

$configJson = Join-Path $curryDir "config.json"
$projectName = (Split-Path $ProjectDir -Leaf) -replace '[^a-zA-Z0-9_-]', '_'

if (-not $ServerName) {
    $ServerName = "curry-$projectName"
}
$toolPrefix = "curry_$($projectName -replace '-','_')"

if (-not (Test-Path $configJson)) {
    $coreDbForward = $CORE_DB -replace '\\', '/'
    $localDbRel    = ".curry/curry.db"

    $cfg = [ordered]@{
        project             = $projectName
        version             = 1
        core_db             = $coreDbForward
        local_db            = $localDbRel
        default_model       = "claude-sonnet-4-6"
        default_model_version = 1
        mcp_tool_prefix     = $toolPrefix
    }
    $cfg | ConvertTo-Json -Depth 5 | Set-Content $configJson -Encoding UTF8
    Write-OK "Created: $configJson"
} else {
    Write-OK "Exists:  $configJson"
    $existingCfg = Get-Content $configJson -Raw | ConvertFrom-Json
    Write-OK "Project: $($existingCfg.project)  prefix: $($existingCfg.mcp_tool_prefix)"
    # Use the prefix already stored in config
    $toolPrefix = $existingCfg.mcp_tool_prefix
}

# ─── 5. Merge into claude_desktop_config.json ─────────────────────────────────

Write-Step "Updating Claude desktop config"
Write-Host "   Config path: $CLAUDE_CFG"

# Load existing config or start fresh
if (Test-Path $CLAUDE_CFG) {
    $raw     = Get-Content $CLAUDE_CFG -Raw -Encoding UTF8
    $desktop = $raw | ConvertFrom-Json
} else {
    $claudeDir = Split-Path $CLAUDE_CFG
    if (-not (Test-Path $claudeDir)) {
        New-Item -ItemType Directory -Path $claudeDir -Force | Out-Null
    }
    $desktop = [PSCustomObject]@{ mcpServers = [PSCustomObject]@{} }
}

# Ensure mcpServers key exists
if (-not ($desktop | Get-Member -Name mcpServers -MemberType NoteProperty)) {
    $desktop | Add-Member -NotePropertyName mcpServers -NotePropertyValue ([PSCustomObject]@{})
}

# Check for collision
$existingEntry = $desktop.mcpServers | Get-Member -Name $ServerName -MemberType NoteProperty
if ($existingEntry -and -not $Force) {
    Write-Warn "An entry named '$ServerName' already exists in mcpServers."
    Write-Warn "Use -Force to overwrite it, or supply a different -ServerName."
    Write-Fail "Aborting to avoid clobbering existing config."
}

# Build the new entry
$newEntry = [PSCustomObject]@{
    command = $pythonExe
    args    = @(
        $MCP_SERVER,
        "--project",
        $ProjectDir
    )
}

# Add or replace
if ($existingEntry) {
    $desktop.mcpServers.$ServerName = $newEntry
    Write-Warn "Replaced existing entry: $ServerName"
} else {
    $desktop.mcpServers | Add-Member -NotePropertyName $ServerName -NotePropertyValue $newEntry
    Write-OK "Added entry: $ServerName"
}

# Write back with 2-space indent to match Claude's own formatting
$desktop | ConvertTo-Json -Depth 10 | Set-Content $CLAUDE_CFG -Encoding UTF8
Write-OK "Saved: $CLAUDE_CFG"

# ─── 6. Summary ───────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Curry wired successfully." -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Server name  : $ServerName"
Write-Host "  Project dir  : $ProjectDir"
Write-Host "  Tool prefix  : $toolPrefix"
Write-Host "  Python       : $pythonExe"
Write-Host ""
Write-Host "  Tools available in Claude (22 static + 1 per registered function):"
Write-Host "    ${toolPrefix}_session_info      -- verify wiring"
Write-Host "    ${toolPrefix}_integrity_check   -- DB health"
Write-Host "    ${toolPrefix}_declare_constant  -- declare a constant"
Write-Host "    ${toolPrefix}_declare_function  -- declare a function"
Write-Host "    ${toolPrefix}_call_function     -- execute a function"
Write-Host "    ${toolPrefix}_record_inference  -- record an LLM call"
Write-Host "    ... and 16 more."
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Yellow
Write-Host "   1. Restart Claude desktop completely (Quit from tray, reopen)."

if (-not (Test-Path $CORE_DB)) {
    Write-Host "   2. Register a model in the core DB first:" -ForegroundColor Yellow
    Write-Host "        python `"$MCP_SERVER`" --help  (to verify the server starts)" -ForegroundColor Yellow
    Write-Host "        -- then in Python:" -ForegroundColor Yellow
    Write-Host "        from curry_core import Curry" -ForegroundColor DarkGray
    Write-Host "        db = Curry(r'$CORE_DB')" -ForegroundColor DarkGray
    Write-Host "        db.register_model('claude-sonnet-4-6', 1, 'release_20260101'," -ForegroundColor DarkGray
    Write-Host "                          temperature=1.0, top_p=0.999, max_tokens=8192)" -ForegroundColor DarkGray
} else {
    Write-Host "   2. In Claude, ask: 'Call ${toolPrefix}_session_info' to verify the connection." -ForegroundColor Yellow
}

Write-Host ""
