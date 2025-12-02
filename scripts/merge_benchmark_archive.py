#!/usr/bin/env python3
"""将 benchmark_archive 中的文件按目录结构合并到 benchmarks 目录."""

import shutil
from pathlib import Path


def merge_benchmark_archive(
    source_dir: Path,
    target_dir: Path,
    dry_run: bool = False,
) -> None:
    """将 source_dir 下的文件按目录结构复制到 target_dir.

    Args:
        source_dir: 源目录 (benchmark_archive)
        target_dir: 目标目录 (benchmarks)
        dry_run: 若为 True，仅打印操作而不实际复制
    """
    if not source_dir.exists():
        print(f"源目录不存在: {source_dir}")
        return

    if not target_dir.exists():
        print(f"目标目录不存在: {target_dir}")
        return

    # 遍历 source_dir 下的所有子目录
    for sub_dir in sorted(source_dir.iterdir()):
        if not sub_dir.is_dir():
            continue

        target_sub_dir = target_dir / sub_dir.name
        if not target_sub_dir.exists():
            print(f"目标子目录不存在，跳过: {target_sub_dir}")
            continue

        # 遍历子目录下的所有文件
        for file_path in sub_dir.iterdir():
            if not file_path.is_file():
                continue

            target_file = target_sub_dir / file_path.name
            action = "覆盖" if target_file.exists() else "复制"

            if dry_run:
                print(f"[DRY RUN] {action}: {file_path} -> {target_file}")
            else:
                shutil.copy2(file_path, target_file)
                print(f"{action}: {file_path.name} -> {target_sub_dir.name}/")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="将 benchmark_archive 中的文件合并到 benchmarks 目录"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印操作，不实际复制文件",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).parent.parent / "benchmark_archive",
        help="源目录路径 (默认: benchmark_archive)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).parent.parent / "benchmarks",
        help="目标目录路径 (默认: benchmarks)",
    )
    args = parser.parse_args()

    print(f"源目录: {args.source.resolve()}")
    print(f"目标目录: {args.target.resolve()}")
    print("-" * 50)

    merge_benchmark_archive(args.source, args.target, dry_run=args.dry_run)

    print("-" * 50)
    print("完成!")


if __name__ == "__main__":
    main()

