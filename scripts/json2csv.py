#!/usr/bin/env python

import gzip
import json
import sys
from pathlib import Path


def json2csv(data_dir: Path) -> None:
    for path in sorted(data_dir.glob("x??.json.gz")):
        out_path = path.parent / (path.name[:3] + ".csv")
        if not out_path.exists():
            json2csv_file(path, out_path)


def json2csv_file(in_path: Path, out_path: Path) -> None:
    print(f"Writing {out_path}")
    with open(out_path, "w") as out:
        for line in gzip.open(in_path):
            ent = json.loads(line)
            ident = ent["id"]
            try:
                desc = ent["labels"]["en"]["value"]
            except KeyError:
                continue
            out.write(f"{ident},label,{desc}\n")
            claims = ent["claims"]
            for prop, lst in claims.items():
                for cp in lst:
                    try:
                        val = cp["mainsnak"]["datavalue"]["value"]
                        if isinstance(val, dict):
                            if "id" in val:
                                val = val["id"]
                            elif "time" in val:
                                val = val["time"]
                            else:
                                continue
                            out.write(f"{ident},{prop},{val}\n")
                    except KeyError:
                        pass


if __name__ == "__main__":
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    json2csv(data_dir)
