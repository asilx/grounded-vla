import argparse

from grounded_vla.training.data import prepare_dataset


def main():
    p = argparse.ArgumentParser(
        description="Validate causal episodes and compute train-only statistics"
    )
    p.add_argument("manifest")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = prepare_dataset(args.manifest, args.out)
    print(f"Prepared {len(result['episodes'])} episodes: {args.out}")


if __name__ == "__main__":
    main()
