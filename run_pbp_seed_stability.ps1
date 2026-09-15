$ErrorActionPreference = 'Stop'

$projectRoot = 'H:\WSY\ReID\VSLA-CLIP-master'
$condaExe = 'D:\anaconda\Scripts\conda.exe'
$stage1Checkpoint = 'output/G2A/vifi_pbp_final_lr2e4/ViT-B-16_stage1_120.pth'
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
    Write-Host "[$(Get-Date -Format s)] START $($run.Name)"
    & $condaExe run --no-capture-output -n ReID python train_reidadapter.py `
        --config_file $run.Config `
        --stage1weight $stage1Checkpoint `
        SOLVER.SEED $run.Seed `
        OUTPUT_DIR $run.Output

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed: $($run.Name) (exit code $LASTEXITCODE)"
    }
    Write-Host "[$(Get-Date -Format s)] FINISH $($run.Name)"
}
