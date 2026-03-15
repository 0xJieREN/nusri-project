from nusri_project.strategy.label_optimization_round1 import (
    build_round1_horizons,
    build_round1_matrix,
    build_round1_trading_shells,
)


def main() -> int:
    print("label_horizons:", build_round1_horizons())
    print("shells:", sorted(build_round1_trading_shells().keys()))
    print("matrix_size:", len(build_round1_matrix()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
