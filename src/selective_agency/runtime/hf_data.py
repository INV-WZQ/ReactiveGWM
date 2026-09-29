"""Translate the released tar-member index into the portable sample CSV."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from .hub import DATASET_REPO
from .io import relative_path, write_json


def member_path(root, name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"invalid archive member: {name}")
    destination = (Path(root) / str(path)).resolve()
    if not destination.is_relative_to(Path(root).resolve()):
        raise ValueError(f"archive member escapes destination: {name}")
    return destination


def extract_selected(archive, destination, names):
    """Extract regular files only; never follow archive links or overwrite assets."""
    remaining = set(names)
    with tarfile.open(archive, "r:") as handle:
        for item in handle:
            if item.name not in remaining:
                continue
            target = member_path(destination, item.name)
            if not item.isfile():
                raise ValueError(f"expected a regular file: {item.name}")
            if target.exists():
                if target.stat().st_size != item.size:
                    raise ValueError(f"existing extracted asset differs: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".incomplete")
                with handle.extractfile(item) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                temporary.replace(target)
            remaining.remove(item.name)
            if not remaining:
                break
    if remaining:
        raise FileNotFoundError(f"members missing from {archive}: {sorted(remaining)[:5]}")


def archive_plan(rows, subset, include_video=False):
    plan = defaultdict(set)
    for row in rows:
        names = json.loads(row["masks_members_json"])
        names += [row.get(k, "") for k in ("first_frame_member", "controls_member", "latents_member")]
        if include_video:
            names.append(row["video_member"])
        plan[row["archive"]].update(n for n in names if n)
        if not row.get("controls_member"):
            plan[row["annotation_archive"]].add(row["annotation_member"])
        text_archive = row.get("text_cache_archive") or f"{subset}/shared/part-00001.tar"
        text_member = row.get("text_cache_member") or f"{subset}/cache/pretraining/control_text.safetensors"
        plan[text_archive].update([text_member, str(PurePosixPath(text_member).with_suffix(".json"))])
    return dict(plan)


def native_controls(row, root, subset, annotations):
    """Read only model_input fields via the original native-format parsers."""
    annotation = member_path(root, row["annotation_member"])
    if annotation not in annotations:
        if annotation.suffix == ".parquet":
            import pyarrow.parquet as pq

            annotations[annotation] = pq.read_table(annotation).to_pylist()
        else:
            annotations[annotation] = annotation.read_text(encoding="utf-8").splitlines()
    source = annotations[annotation][int(row["annotation_row"])]
    if subset == "HNM":
        from selective_agency.data.hnm import parse_hnm_record

        spec = json.loads((root / "HNM/cache/pretraining/control_text.json").read_text())["spec"]
        record = parse_hnm_record(
            source, asset_root=root / "HNM/dataset" / row["split"],
            action_vocabulary=spec["actions"], expected_split=row["split"], require_assets=False,
        )
    else:
        from selective_agency.data.sf3 import SF3ActionSpaces, parse_sf3_record

        record = parse_sf3_record(
            source, repo_root={
                "sf3_legacy_npc0": root / "SF3/dataset/npc0",
                "sf3_v2_npc1": root / "SF3/dataset/npc1",
            }, action_spaces=SF3ActionSpaces(), expected_split=row["split"], require_assets=False,
        )
    if record.sample_id != row["sample_id"]:
        raise ValueError("HF sample ID differs from its native annotation row")
    expected = [member_path(root, name) for name in json.loads(row["masks_members_json"])]
    if list(record.x0_masks) != expected:
        raise ValueError("HF mask order differs from native subject order")
    subjects = []
    for slot, name in enumerate(record.subjects):
        item = {"name": name}
        if record.control_kind[slot] == 1:
            item["actions"] = [step[slot] for step in record.actions]
        else:
            item["prompt"] = record.npc_prompts[slot]
        subjects.append(item)
    return {
        "subjects": subjects, "sample_index": record.sample_index,
        "family_id": record.family_id, "branch": record.branch,
        "source_identity": record.source_identity, "source_group_identity": record.source_group_identity,
    }


def build_csv(rows, root, output, subset, *, include_video=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"choose a new CSV destination: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    records, annotations = [], {}
    for index, row in enumerate(rows):
        controls = row.get("controls_member")
        if controls:
            controls_path = member_path(root, controls)
        else:
            controls_path = output.parent / (output.stem + "_controls") / f"{index:06d}.json"
            if controls_path.exists():
                raise FileExistsError(controls_path)
            write_json(controls_path, native_controls(row, root, subset, annotations))
        text = row.get("text_cache_member") or f"{subset}/cache/pretraining/control_text.safetensors"
        def location(name):
            if not name:
                return ""
            path = member_path(root, name)
            if not path.is_file():
                raise FileNotFoundError(path)
            return relative_path(path, output.parent)
        records.append({
            "sample_id": row["sample_id"], "split": row["split"],
            "video": location(row["video_member"]) if include_video else "",
            "first_frame": location(row["first_frame_member"]),
            "masks": json.dumps([location(n) for n in json.loads(row["masks_members_json"])]),
            "controls": relative_path(controls_path, output.parent),
            "latents": location(row.get("latents_member", "")), "text_cache": location(text),
        })
    temporary = output.with_suffix(".csv.incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    from .csv_data import CSVIndex

    CSVIndex(temporary)  # Check all control lengths, text mappings and subject counts.
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, choices=("hnm", "sf3"))
    parser.add_argument("--split", nargs="+", choices=("train", "val", "validation", "test"), default=["val"])
    parser.add_argument("--sample-id", nargs="+", help="select exact HF sample IDs")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=2, help="samples per split (default: 2)")
    selection.add_argument("--all", action="store_true", help="select all rows of requested splits")
    parser.add_argument("--repo-id", default=DATASET_REPO)
    parser.add_argument("--revision", default="main", help="HF dataset branch or tag (default: main)")
    parser.add_argument("--output", required=True, help="new prepared dataset directory")
    parser.add_argument("--local-repo", help="use an already downloaded HF repository")
    parser.add_argument("--include-video", action="store_true", help="also extract GT RGB videos for cache preparation")
    parser.add_argument("--dry-run", action="store_true", help="read indices and print required archive sizes only")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    subset = "HNM" if args.game == "hnm" else "SF3"
    output = Path(args.output).expanduser().resolve()
    if (output / "samples.csv").exists():
        parser.error("output already contains samples.csv; choose a new directory")
    if args.local_repo:
        local = Path(args.local_repo).expanduser().resolve()
        def fetch(name):
            return member_path(local, name).resolve(strict=True)
    else:
        from huggingface_hub import hf_hub_download

        def fetch(name):
            return Path(hf_hub_download(
                args.repo_id, name, repo_type="dataset", revision=args.revision,
                local_dir=output / "downloads",
            ))
    rows = []
    for split in dict.fromkeys("val" if x == "validation" else x for x in args.split):
        with fetch(f"{subset}/sample_index_{split}.csv").open(newline="", encoding="utf-8") as handle:
            selected = list(csv.DictReader(handle))
        if args.sample_id:
            selected = [r for r in selected if r["sample_id"] in args.sample_id]
        elif not args.all:
            selected = selected[:args.limit]
        rows.extend(selected)
    if not rows or (args.sample_id and set(args.sample_id) != {r["sample_id"] for r in rows}):
        parser.error("no samples selected or sample ID absent from requested splits")
    plan = archive_plan(rows, subset, args.include_video)
    with fetch("volume_manifest.csv").open(newline="", encoding="utf-8") as handle:
        volumes = {r["archive"]: int(r["tar_bytes"]) for r in csv.DictReader(handle)}
    report = {
        "repo_id": args.repo_id, "revision": args.revision, "game": args.game,
        "samples": len(rows), "archives": {name: volumes[name] for name in sorted(plan)},
        "download_bytes": sum(volumes[name] for name in plan),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.dry_run:
        return
    for archive, members in sorted(plan.items()):
        print(f"Preparing {archive}", flush=True)
        extract_selected(fetch(archive), output / "assets", members)
    build_csv(rows, output / "assets", output / "samples.csv", subset, include_video=args.include_video)
    write_json(output / "source.json", report)
    print(f"Prepared {output / 'samples.csv'}", flush=True)
