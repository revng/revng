#!/usr/bin/env python3

"""
Generate a build.ninja to compile win32metadata partitions into PDBs.

Usage:

    # Generate build.ninja
    python3 compile-to-pdb.py [options] [PARTITION ...]

    # Build all PDBs
    ninja -C build

    # Build a single PDB
    ninja -C build Memory.pdb

Compiler flags are read from the win32metadata .rsp files.
"""

import argparse
from pathlib import Path
import sys


def log(message: str) -> None:
    print(message, file=sys.stderr)


def parse_rsp_section(rsp_path, section):
    """Extract lines from a named --section of a ClangSharp .rsp file."""
    lines = []
    in_section = False
    for line in Path(rsp_path).read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == f"--{section}":
            in_section = True
        elif stripped.startswith("--"):
            in_section = False
        elif in_section:
            lines.append(stripped)
    return lines


def parse_rsp_traverse_files(rsp_path, include_root, partition_dir):
    """Extract --traverse file paths from a ClangSharp .rsp file.

    Resolves <IncludeRoot> and <PartitionDir> placeholders to absolute paths.
    """
    return [
        line.replace("<IncludeRoot>", str(include_root))
            .replace("<PartitionDir>", str(partition_dir))
        for line in parse_rsp_section(rsp_path, "traverse")
    ]


def discover_partitions(partitions_dir):
    """Return sorted list of partition names that have a main.cpp."""
    return sorted(
        p.name for p in partitions_dir.iterdir()
        if (p / "main.cpp").is_file()
    )


def ninja_escape(s):
    """Escape a string for ninja build file syntax."""
    return str(s).replace("$", "$$").replace(" ", "$ ").replace(":", "$:")


def main():
    parser = argparse.ArgumentParser(
        description="Generate build.ninja for win32metadata PDB compilation.",
    )
    parser.add_argument(
        "partitions", nargs="*",
        help="Partition names to include (default: all)",
    )
    parser.add_argument(
        "--win32meta-root",
        help="Path to win32metadata repo root",
    )
    parser.add_argument(
        "--output-dir", "-o", default="build",
        help="Output directory for build.ninja and PDBs (default: build)",
    )
    parser.add_argument(
        "--clang",
        help="Path to patched clang binary",
    )
    parser.add_argument(
        "--lld-link", default="lld-link",
        help="lld-link command (default: lld-link)",
    )
    parser.add_argument(
        "--vc19-include",
        help="Path to VC19 CRT include directory",
    )
    parser.add_argument(
        "--target-triple",
        help="Clang target triple",
    )
    parser.add_argument(
        "--arch-rsp", action="append", default=[],
        help="Architecture-specific .rsp file name (repeatable, "
             "default: baseSettings.x64.rsp)",
    )
    args = parser.parse_args()

    # Derive paths from win32metadata root
    root = Path(args.win32meta_root).resolve()
    partitions_dir = root / "generation" / "WinSDK" / "Partitions"
    sdk_inc_root = root / "generation" / "WinSDK" / "RecompiledIdlHeaders"
    additional_inc = root / "generation" / "WinSDK" / "inc"
    scraper_dir = root / "sources" / "GeneratorSdk" / "tools" / "assets" / "scraper"
    base_rsp = scraper_dir / "baseSettings.rsp"

    arch_rsp_paths = [scraper_dir / name for name in args.arch_rsp]

    required = [
        (partitions_dir, "Partitions directory"),
        (base_rsp, "baseSettings.rsp"),
    ] + [(p, p.name) for p in arch_rsp_paths]

    for path, label in required:
        if not path.exists():
            log(f"error: {label} not found: {path}")
            return 1

    # Parse compiler flags from .rsp files
    clang_flags = parse_rsp_section(base_rsp, "additional")
    for p in arch_rsp_paths:
        clang_flags += parse_rsp_section(p, "additional")

    include_dirs = [
        additional_inc,
        Path(args.vc19_include).resolve(),
        sdk_inc_root / "shared",
        sdk_inc_root / "um",
        sdk_inc_root / "ucrt",
        sdk_inc_root / "winrt",
    ]

    # Also add any subdirectories (cpdk, gl, alljoyn_c, etc.)
    for root_sub in include_dirs[-4:]:
        include_dirs += sorted(p for p in root_sub.iterdir() if p.is_dir())

    # Discover / validate partitions
    all_partitions = discover_partitions(partitions_dir)
    if args.partitions:
        unknown = set(args.partitions) - set(all_partitions)
        if unknown:
            log(f"error: unknown partition(s): {', '.join(sorted(unknown))}")
            return 1
        partitions = args.partitions
    else:
        partitions = all_partitions

    # Prepare output directories
    output_dir = Path(args.output_dir).resolve()
    obj_dir = output_dir / "obj"
    obj_dir.mkdir(parents=True, exist_ok=True)

    clang_bin = Path(args.clang).resolve()
    lld_link = args.lld_link

    # Write shared clang response file.
    # Using a response file (@file) avoids shell-escaping issues with flags
    # like -Wno-#pragma-messages.
    rsp_file = obj_dir / "_compile.rsp"
    rsp_file.write_text("\n".join([
        f"--target={args.target_triple}",
        "-gcodeview",
        "-g",
        "-fno-eliminate-unused-debug-types",
        "-fcase-insensitive-paths",
        "-c",
        *(f"-I{d}" for d in include_dirs),
        *clang_flags,
        # Suppress remaining warnings — we only care about debug info.
        "-w",
    ]) + "\n")

    # Generate build.ninja
    ninja_path = output_dir / "build.ninja"
    esc = ninja_escape

    with open(ninja_path, "w") as f:
        f.write(f"# Generated by compile-to-pdb.py — {len(partitions)} partitions\n\n")

        # Variables (avoid "rspfile" — it's reserved by ninja)
        f.write(f"clang = {esc(clang_bin)}\n")
        f.write(f"lld_link = {esc(lld_link)}\n")
        f.write(f"compile_rsp = {esc(rsp_file)}\n\n")

        # Rules
        f.write("rule cc\n")
        f.write("  command = $clang @$compile_rsp $allowed_flags"
                " -o $out $in\n")
        f.write("  description = CC $partition\n\n")

        # Link .obj into a dummy DLL just to produce the PDB.
        # -dll -noentry       → no entry point needed
        # -nodefaultlib       → no import libraries needed
        # -force:unresolved   → ignore missing symbols (e.g. GUID_NULL)
        f.write("rule link\n")
        f.write("  command = $lld_link -dll -noentry -nodefaultlib"
                " -force:unresolved -debug -pdb:$pdb -out:$out $in"
                " 2>&1 | grep -v -e '^lld-link' -e '^>>>' || true\n")
        f.write("  description = PDB $partition\n\n")

        # Per-partition targets
        pdb_aliases = []
        for name in partitions:
            part_dir = partitions_dir / name
            allowed = parse_rsp_traverse_files(
                part_dir / "settings.rsp", sdk_inc_root, part_dir,
            )

            cpp_e = esc(part_dir / "main.cpp")
            obj_e = esc(obj_dir / f"{name}.obj")
            dll_e = esc(obj_dir / f"{name}.dll")
            pdb_e = esc(output_dir / f"{name}.pdb")

            allowed_flags = " ".join(
                f"-fdebug-info-allowed-file={esc(p)}" for p in allowed
            )

            f.write(f"build {obj_e}: cc {cpp_e}\n")
            f.write(f"  partition = {name}\n")
            f.write(f"  allowed_flags = {allowed_flags}\n")

            f.write(f"build {dll_e} | {pdb_e}: link {obj_e}\n")
            f.write(f"  pdb = {pdb_e}\n")
            f.write(f"  partition = {name}\n")

            f.write(f"build {name}.pdb: phony {pdb_e}\n\n")
            pdb_aliases.append(f"{name}.pdb")

        # Default target
        f.write(f"build all: phony {' '.join(pdb_aliases)}\n")
        f.write("default all\n")

    log(f"wrote {ninja_path}  ({len(partitions)} partitions)")
    log(f"  rsp: {rsp_file}")
    log(f"  flags from: {base_rsp}")
    for p in arch_rsp_paths:
        log(f"             {p}")
    log(f"\nrun:  ninja -C {output_dir}")
    log(f"      ninja -C {output_dir} Memory.pdb        # single partition")
    log(f"      ninja -C {output_dir} -j$(nproc)        # parallel")


if __name__ == "__main__":
    main()
