# Regime Trader - PowerShell Launcher Menu
# Run: .\menu\regime_trader.ps1

$ROOT = Resolve-Path "$PSScriptRoot\.."
Set-Location $ROOT

# -- Colors and Styling --
$cyan    = "$([char]27)[1;36m"
$yellow  = "$([char]27)[1;33m"
$green   = "$([char]27)[1;32m"
$blue    = "$([char]27)[1;34m"
$magenta = "$([char]27)[1;35m"
$red     = "$([char]27)[1;31m"
$dim     = "$([char]27)[2m"
$reset   = "$([char]27)[0m"

# -- State --
$assetGroup = "stocks"
$python = "python" # adjust to 'py -3.12' if needed

function Print-Header {
    Clear-Host
    Write-Host "${cyan}  +--------------------------------------+"
    Write-Host "  |         REGIME  TRADER               |"
    Write-Host "  |   HMM Volatility Regime System       |"
    Write-Host "  +--------------------------------------+${reset}"
    Write-Host "  ${dim}Working directory: $ROOT${reset}"
    Write-Host "  ${yellow}Active group : ${cyan}$assetGroup${reset}  ${dim}(change with g)${reset}"
    Write-Host ""
}

function Print-Menu {
    Write-Host "  ${yellow}-- Trading ---------------------------------${reset}"
    Write-Host "  ${green}[1]${reset}  Train HMM        ${dim}(fetch latest bars, fit model, save)${reset}"
    Write-Host "  ${green}[2]${reset}  Dry Run          ${dim}(full pipeline - no orders placed)${reset}"
    Write-Host "  ${green}[3]${reset}  Live / Paper     ${dim}(start trading loop)${reset}"
    Write-Host ""
    Write-Host "  ${yellow}-- Backtesting -----------------------------${reset}"
    Write-Host "  ${green}[0]${reset}  Full Cycle       ${dim}(HMM + backtest ALL 3 groups, summary table)${reset}"
    Write-Host "  ${green}[4]${reset}  Backtest Quick   ${dim}(active group, 2020-now, no benchmark)${reset}"
    Write-Host "  ${green}[5]${reset}  Backtest Group   ${dim}(active group, 2020-now, benchmark)${reset}"
    Write-Host "  ${green}[6]${reset}  Forward Test     ${dim}(hold-out 2024-today, out-of-sample)${reset}"
    Write-Host ""
    Write-Host "  ${green}[8]${reset}  Train + Backtest ${dim}(retrain HMM then full benchmark, active group)${reset}"
    Write-Host ""
    Write-Host "  ${yellow}-- Optimisation ----------------------------${reset}"
    Write-Host "  ${green}[7]${reset}  Param Sweep      ${dim}(tune on 2020-2023, active group)${reset}"
    Write-Host "  ${green}[9]${reset}  Rolling WFO      ${dim}(12m tune / 3m test, 4:1 ratio, robustness check)${reset}"
    Write-Host "  ${green}[w]${reset}  WFO Windows      ${dim}(show fold schedule and bar counts - no backtests run)${reset}"
    Write-Host ""
    Write-Host "  ${yellow}-- Asset Groups ----------------------------${reset}"
    Write-Host "  ${blue}[g]${reset}  Change Group     ${dim}(stocks | crypto | indices)${reset}"
    Write-Host ""
    Write-Host "  ${red}[q]${reset}  Quit"
    Write-Host ""
}

function Run-Command($label, $cmd) {
    Write-Host ""
    Write-Host "  ${cyan}>> $label${reset}"
    Write-Host "  ${dim}$cmd${reset}"
    Write-Host "  ${dim}-----------------------------------------${reset}"
    Write-Host ""
    Invoke-Expression $cmd
    $exitCode = $LASTEXITCODE
    Write-Host ""
    if ($exitCode -eq 0) {
        Write-Host "  ${green}Done (exit 0)${reset}"
    } else {
        Write-Host "  ${red}Exited with code $exitCode${reset}"
    }
    Write-Host ""
    Read-Host "  Press Enter to return to menu..."
}

function Select-Group {
    Write-Host ""
    Write-Host "  ${yellow}Select asset group:${reset}"
    Write-Host "  ${blue}[1]${reset}  stocks   ${dim}(SPY QQQ AAPL MSFT AMZN GOOGL NVDA META TSLA AMD)${reset}"
    Write-Host "  ${blue}[2]${reset}  crypto   ${dim}(BTC ETH SOL AVAX DOGE LTC LINK UNI)${reset}"
    Write-Host "  ${blue}[3]${reset}  indices  ${dim}(SPY QQQ DIA IWM GLD TLT EFA EEM VNQ USO)${reset}"
    Write-Host ""
    $gchoice = Read-Host "  Your choice"
    switch ($gchoice) {
        "1" { $script:assetGroup = "stocks" }
        "2" { $script:assetGroup = "crypto" }
        "3" { $script:assetGroup = "indices" }
        default { Write-Host "  ${red}Invalid - keeping '$assetGroup'${reset}"; Start-Sleep -Seconds 1 }
    }
}

while ($true) {
    Print-Header
    Print-Menu
    $choice = Read-Host "  Your choice"

    switch ($choice) {
        "0" {
            Run-Command "Full Cycle - HMM + Backtest for ALL 3 groups" "$python main.py full-cycle --start 2020-01-01"
        }
        "1" {
            Run-Command "Train HMM - group: $assetGroup" "$python main.py trade --train-only --asset-group $assetGroup"
        }
        "2" {
            Run-Command "Dry Run - group: $assetGroup" "$python main.py trade --dry-run --asset-group $assetGroup"
        }
        "3" {
            Run-Command "Live / Paper Trade - group: $assetGroup" "$python main.py trade --asset-group $assetGroup"
        }
        "4" {
            Run-Command "Backtest group: $assetGroup 2020-now (no benchmark)" "$python main.py backtest --asset-group $assetGroup --start 2020-01-01"
        }
        "5" {
            Run-Command "Backtest group: $assetGroup 2020-now (benchmark)" "$python main.py backtest --asset-group $assetGroup --start 2020-01-01 --compare"
        }
        "6" {
            Run-Command "Forward Test - group: $assetGroup 2024-today" "$python main.py backtest --asset-group $assetGroup --start 2024-01-01 --compare"
        }
        "7" {
            Run-Command "Parameter Sweep - group: $assetGroup" "$python tools/param_sweep.py --asset-group $assetGroup --start 2020-01-01 --end 2023-12-31"
        }
        "8" {
            Run-Command "Train HMM + Backtest - group: $assetGroup" "$python main.py trade --train-only --asset-group $assetGroup; if (`$LASTEXITCODE -eq 0) { $python main.py backtest --asset-group $assetGroup --start 2020-01-01 --compare }"
        }
        "9" {
            Run-Command "Rolling WFO - group: $assetGroup" "$python tools/rolling_wfo.py --asset-group $assetGroup --start 2020-01-01 --tune-months 12 --test-months 3 --step-months 3"
        }
        "g" {
            Select-Group
        }
        "w" {
            Run-Command "WFO Window Plan - group: $assetGroup" "$python tools/rolling_wfo.py --asset-group $assetGroup --start 2020-01-01 --tune-months 12 --test-months 3 --step-months 3 --show-windows"
        }
        "q" {
            Write-Host "  ${dim}Goodbye.${reset}"
            exit 0
        }
        default {
            Write-Host "  ${red}Invalid option '$choice' - try again.${reset}"
            Start-Sleep -Seconds 1
        }
    }
}
