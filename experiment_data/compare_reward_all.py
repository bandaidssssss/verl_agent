"""Compare reward curves for the base run and four dated agent runs."""

from pathlib import Path

try:
    from .compare_reward import parse_log, plot_time_token_curves
except ImportError:
    # Support running this file directly: python experiment_data/compare_reward_all.py
    from compare_reward import parse_log, plot_time_token_curves


SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
OUTPUT_FILE = SCRIPT_DIR / "result" / "base_vs_0731_0807_0814_0819.png"
MAX_STEP = 134

LOG_FILES = {
    "base": LOG_DIR / "base_1024_4096_lr_1e-6_n5_2epochs.log",
    "0731": LOG_DIR / "0731agent.log",
    "0807": LOG_DIR / "0807agent.log",
    "0814": LOG_DIR / "0814agent.log",
    "0819": LOG_DIR / "0819_optimized.log",
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
