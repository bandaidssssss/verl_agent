"""Compare reward curves for the base run and four dated agent runs."""

from pathlib import Path

try:
    from .compare_reward import parse_log, plot_time_token_curves
except ImportError:
    # Support running this file directly: python experiment_data/compare_reward_all.py
    from compare_reward import parse_log, plot_time_token_curves


SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / ".." / "output" / "0901_1346_2026" / "trials"
OUTPUT_FILE = SCRIPT_DIR / "0901_1346_2026" / "reward_metrics_all.png"
MAX_STEP = 134

LOG_FILES = {
    # "base": "/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/ssh_agent/experiment_data/math_logs/0q6_base.log",

    "0001": LOG_DIR / "0011" / "train.log",
    # "entropy0001":"experiment_data/logs/entropy0001.log",
    # "entropy0003":"experiment_data/logs/entropy0003.log",
    # "0002": LOG_DIR / "0012" / "train.log",
    # "0003": LOG_DIR / "0013" / "train.log",
    # "0004": LOG_DIR / "0014" / "train.log",
    # "manual_01": "/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/verl-0.7/saves/qwen3_0q6_math/2026.09.01-16.35.log",
    # "manual_02": "/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/verl-0.7/saves/qwen3_0q6_math/2026.09.02-09.15.log",
    # "manual_03": "/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/verl-0.7/saves/qwen3_0q6_math/2026.09.02-15.55.log",
    "manual_04": "/mnt/gxnbdrw-2/gxnbdrw-2/wangxinyuan/verl-0.7/saves/qwen3_0q6_math/2026.09.03-09.43.log",
    # "0005": LOG_DIR / "0015" / "train.log",
    # "0006": LOG_DIR / "0016" / "train.log",
    # "0007": LOG_DIR / "0017" / "train.log",
    "0008": LOG_DIR / "0018" / "train.log",
    # "0009": LOG_DIR / "0019" / "train.log",
    # "0010": LOG_DIR / "0020" / "train.log",
}



def main():
    all_data = {}

    for label, log_file in LOG_FILES.items():
        try:
            data = parse_log(log_file, max_step=MAX_STEP)
        except FileNotFoundError:
            print(f"错误：文件 {log_file} 不存在，跳过。")
            continue

        if not data:
            print(
                f"警告：{log_file} 中缺少必要字段 "
                "（rewards / time_per_step / total_num_tokens），跳过。"
            )
            continue

        all_data[label] = data
        print(
            f"{label}: 提取了 {len(data)} 步 "
            f"(step {min(data)} ~ {max(data)})"
        )

    if not all_data:
        print("没有有效数据，请检查日志文件路径和字段是否存在。")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plot_time_token_curves(all_data, output_file=str(OUTPUT_FILE))


if __name__ == "__main__":
    main()
