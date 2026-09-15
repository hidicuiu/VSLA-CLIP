$ErrorActionPreference = 'Stop'

$projectRoot = 'H:\WSY\ReID\VSLA-CLIP-master'
$condaExe = 'D:\anaconda\Scripts\conda.exe'
$fullConfig = 'configs/adapter/vit_adapter_g2a_vifi_pbp_train_LR2E-4.yml'
$noPbpConfig = 'configs/adapter/vit_adapter_g2a_vifi_pbp_no_pbp_lr2e4.yml'

$runs = @(
    @{ Name = 'full_pbp_seed2025'; Config = $fullConfig; Seed = 2025; Output = 'output/G2A/vifi_pbp_seed2025_lr2e4' },
    @{ Name = 'no_pbp_seed2025'; Config = $noPbpConfig; Seed = 2025; Output = 'output/G2A/vifi_no_pbp_seed2025_lr2e4' },
    @{ Name = 'full_pbp_seed2026'; Config = $fullConfig; Seed = 2026; Output = 'output/G2A/vifi_pbp_seed2026_lr2e4' },
    @{ Name = 'no_pbp_seed2026'; Config = $noPbpConfig; Seed = 2026; Output = 'output/G2A/vifi_no_pbp_seed2026_lr2e4' }
)

Set-Location -LiteralPath $projectRoot
foreach ($run in $runs) {
    $weight = Join-Path $run.Output 'ViT-B-16_mAP_best.pth'
    $log = Join-Path $run.Output 'test_log_recheck.txt'
    Write-Host "[$(Get-Date -Format s)] TEST $($run.Name)"
    & $condaExe run --no-capture-output -n ReID python test.py `
        --config_file $run.Config `
        SOLVER.SEED $run.Seed `
        OUTPUT_DIR $run.Output `
        TEST.WEIGHT $weight 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        throw "Test failed: $($run.Name) (exit code $LASTEXITCODE)"
    }
}
