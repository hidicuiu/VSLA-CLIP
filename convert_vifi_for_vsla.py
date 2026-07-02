from collections import OrderedDict
from pathlib import Path

import torch


SOURCE_PATH = Path(
    r"H:\WSY\ReID\VSLA-CLIP-master"
    r"\pretrained\vifi"
    r"\k400_clip_complete_finetuned_30_epochs.pth"
)

OUTPUT_PATH = Path(
    r"H:\WSY\ReID\VSLA-CLIP-master"
    r"\pretrained\vifi"
    r"\vifi_clip_k400_for_vsla.pth"
)


def convert_vifi_checkpoint() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"找不到原始权重：{SOURCE_PATH}")

    print(f"正在读取：{SOURCE_PATH}")
    checkpoint = torch.load(SOURCE_PATH, map_location="cpu")

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("检查点中没有找到 model 参数字典。")

    source_state_dict = checkpoint["model"]

    if not isinstance(source_state_dict, dict):
        raise TypeError("checkpoint['model'] 不是参数字典。")

    converted_state_dict = OrderedDict()
    ignored_keys = []

    for original_key, value in source_state_dict.items():
        key = original_key

        # Remove DistributedDataParallel prefix.
        if key.startswith("module."):
            key = key[len("module."):]

        # Convert ViFi-CLIP visual encoder names to standard CLIP names.
        if key.startswith("image_encoder."):
            key = "visual." + key[len("image_encoder."):]
            converted_state_dict[key] = value

        # Convert ViFi-CLIP text encoder names to standard CLIP names.
        elif key.startswith("text_encoder."):
            key = key[len("text_encoder."):]
            converted_state_dict[key] = value

        # Preserve CLIP logit scale.
        elif key == "logit_scale":
            converted_state_dict[key] = value

        # Kinetics-400 class prompt buffers are not needed by VSLA-CLIP.
        elif key.startswith("prompt_learner."):
            ignored_keys.append(original_key)

        else:
            ignored_keys.append(original_key)

    required_keys = [
        "visual.conv1.weight",
        "visual.class_embedding",
        "visual.positional_embedding",
        "visual.ln_pre.weight",
        "visual.transformer.resblocks.0.attn.in_proj_weight",
        "visual.proj",
        "transformer.resblocks.0.attn.in_proj_weight",
        "positional_embedding",
        "ln_final.weight",
        "text_projection",
        "logit_scale",
    ]

    missing_required_keys = [
        key for key in required_keys
        if key not in converted_state_dict
    ]

    print("\n转换结果：")
    print(f"原始参数数量：{len(source_state_dict)}")
    print(f"保留参数数量：{len(converted_state_dict)}")
    print(f"忽略参数数量：{len(ignored_keys)}")

    print("\n转换后的前20个参数：")
    for key in list(converted_state_dict.keys())[:20]:
        print(key)

    print("\n关键参数检查：")
    for key in required_keys:
        print(f"{key}: {key in converted_state_dict}")

    if missing_required_keys:
        print("\n缺少以下关键参数：")
        for key in missing_required_keys:
            print(key)

        raise RuntimeError("转换后的权重缺少关键CLIP参数，暂不保存。")

    # token_embedding.weight is intentionally absent.
    # VSLA-CLIP restores it from the original OpenAI CLIP checkpoint.
    print(
        "\ntoken_embedding.weight存在：",
        "token_embedding.weight" in converted_state_dict,
    )
    print("该参数显示False是正常的，VSLA-CLIP会从原始CLIP中补回。")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted_state_dict, OUTPUT_PATH)

    print(f"\n转换完成，已保存到：\n{OUTPUT_PATH}")


if __name__ == "__main__":
    convert_vifi_checkpoint()