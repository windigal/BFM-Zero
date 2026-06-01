from __future__ import annotations

import json
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import tyro


BABEL_SPLITS = ("train", "val")

# BABEL feat_p top-level prefix -> HF AMASS raw tarball basename
TAR_MAP = {
    "ACCAD": "ACCAD.tar.bz2",
    "BMLmovi": "BMLmovi.tar.bz2",
    "BMLrub": "BMLrub.tar.bz2",
    "CMU": "CMU.tar.bz2",
    "EKUT": "EKUT.tar.bz2",
    "EyesJapanDataset": "EyesJapanDataset.tar.bz2",
    "HumanEva": "HumanEva.tar.bz2",
    "KIT": "KIT.tar.bz2",
    "MPIHDM05": "HDM05.tar.bz2",
    "DFaust67": "DFaust.tar.bz2",
    "Transitionsmocap": "Transitions.tar.bz2",
    "MPImosh": "MoSh.tar.bz2",
    "TCDhandMocap": "TCDHands.tar.bz2",
    "MPILimits": "PosePrior.tar.bz2",
    "SFU": "SFU.tar.bz2",
    "SSMsynced": "SSM.tar.bz2",
    "TotalCapture": "TotalCapture.tar.bz2",
}


@dataclass
class Args:
    babel_dir: Path = Path("~/dataset/babel_v1-0_release/babel_v1.0_release").expanduser()
    amass_hf_dir: Path = Path("~/dataset/AMASS").expanduser()
    output_dir: Path = Path("~/dataset/AMASS_babel_smplx").expanduser()
    overwrite_output: bool = False


def collect_required_prefixes(babel_dir: Path) -> Counter[str]:
    counter: Counter[str] = Counter()
    for split in BABEL_SPLITS:
        json_path = babel_dir / f"{split}.json"
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for item in payload.values():
            feat_p = item.get("feat_p")
            if not feat_p:
                continue
            prefix = str(feat_p).split("/")[0]
            counter[prefix] += 1
    return counter


def extract_required_amass(args: Args) -> dict[str, object]:
    raw_dir = args.amass_hf_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing AMASS raw dir: {raw_dir}")
    required_prefixes = collect_required_prefixes(args.babel_dir)
    missing_prefixes = [prefix for prefix in required_prefixes if prefix not in TAR_MAP]
    if missing_prefixes:
        raise KeyError(f"No tar mapping for BABEL prefixes: {missing_prefixes}")

    if args.output_dir.exists() and args.overwrite_output:
        for child in args.output_dir.iterdir():
            if child.is_dir():
                for nested in child.iterdir():
                    if nested.is_dir():
                        for item in nested.iterdir():
                            if item.is_file():
                                item.unlink()
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    missing_tars = []
    for prefix in sorted(required_prefixes):
        tar_name = TAR_MAP[prefix]
        tar_path = raw_dir / tar_name
        if not tar_path.exists():
            missing_tars.append(str(tar_path))
            continue
        target_prefix_dir = args.output_dir / prefix
        target_prefix_dir.mkdir(parents=True, exist_ok=True)
        marker = target_prefix_dir / ".extracted"
        if marker.exists():
            extracted.append({"prefix": prefix, "tar": tar_name, "status": "already_present"})
            continue
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(path=target_prefix_dir)
        marker.write_text("ok\n", encoding="utf-8")
        extracted.append({"prefix": prefix, "tar": tar_name, "status": "extracted"})

    summary = {
        "babel_dir": str(args.babel_dir.resolve()),
        "amass_hf_dir": str(args.amass_hf_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "required_prefix_counts": dict(required_prefixes),
        "extracted": extracted,
        "missing_tars": missing_tars,
    }
    (args.output_dir / "extract_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(args: Args) -> None:
    summary = extract_required_amass(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(tyro.cli(Args))
