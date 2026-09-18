"""Download the public PaliGemma tokenizer used by openpi."""

import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="models/paligemma_tokenizer.model")
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    from openpi.shared.download import maybe_download

    source = maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, out)
    print(out)


if __name__ == "__main__":
    main()
