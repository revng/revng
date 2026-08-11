#!/usr/bin/env python3
#
# `push` publishes the transitive derivation closures of Nix flake packages to
# named static binary caches. It creates a Ninja file to pack and sign paths in
# parallel, then uploads NARs before narinfos so incomplete entries are never
# advertised. Unmarked paths use the configured default cache; a structured
# `useCache` derivation attribute selects another cache by name.
#
# Each package's nixpkgs `inputDerivation` is included by default when present.
# Use `--no-input-derivation` to publish only the packages themselves.
#
# Example:
#
#   nix-cache-push push --config cache.yml .#revng .#test/revng
#
# The default mode skips every locally signed path. `--check-remote` instead
# lists remote narinfos and republishes locally signed paths that are missing
# remotely. This is useful after manually purging remote cache entries for
# derivations that remain signed in the local Nix database.
#
# `pack-one` is an internal Ninja worker.
#

import argparse
import base64
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from hashlib import file_digest
from pathlib import Path
from typing import IO

import nacl.signing
import yaml


# Nix's lowercase base32 variant.
NIX32_ALPHABET = "0123456789abcdfghijklmnpqrsvwxyz"

# Ninja invokes this executable for each store path. Under wrapProgram,
# `__file__` points to the wrapped Python program; the wrapper's PATH remains
# inherited by Ninja and its workers.
SELF_PATH = Path(__file__).resolve()


#
# Logging and subprocess helpers
#

def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def echo_cmd(args: list) -> None:
    command = " ".join(shlex.quote(str(argument)) for argument in args)
    print(f"+ {command}", file=sys.stderr, flush=True)


def show_derivation_closure(packages: list[str]) -> dict:
    """Return `nix derivation show -r` JSON for the installables."""
    command = ["nix", "derivation", "show", "-r", *packages]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def query_path_infos(paths: list[str]) -> dict:
    """Return one batched `nix path-info` result for all paths."""
    if not paths:
        return {}
    command = ["nix", "path-info", "--json", "--stdin"]
    result = subprocess.run(
        command,
        input="\n".join(paths),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def package_has_input_derivation(package: str) -> bool:
    """Ask Nix whether an installable exposes inputDerivation."""
    command = [
        "nix",
        "eval",
        "--json",
        "--apply",
        "package: package ? inputDerivation",
        package,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    has_input_derivation = json.loads(result.stdout)
    assert isinstance(has_input_derivation, bool)
    return has_input_derivation


def run_ninja_build(scratch: Path) -> None:
    command = ["ninja", "-C", str(scratch)]
    echo_cmd(command)
    subprocess.run(command, check=True)


def rsync_cache_payload(scratch: Path, remote: str) -> None:
    """Upload NARs and nix-cache-info."""
    command = [
        "rsync",
        "-a",
        "--info=stats1",
        "--ignore-existing",
        "-L",
        "--chmod=D755,F644",
        "--include=/nix-cache-info",
        "--include=/nar/",
        "--include=/nar/*.nar.zst",
        "--exclude=*",
        str(scratch) + "/",
        remote + "/",
    ]
    echo_cmd(command)
    subprocess.run(command, check=True)


def rsync_cache_narinfos(scratch: Path, remote: str) -> None:
    """Upload narinfos after their NARs."""
    command = [
        "rsync",
        "-a",
        "--info=stats1",
        "--ignore-existing",
        "--chmod=D755,F644",
        "--include=/*.narinfo",
        "--exclude=*",
        str(scratch) + "/",
        remote + "/",
    ]
    echo_cmd(command)
    subprocess.run(command, check=True)


def sign_local_store_paths(key_file: str, paths: list[str]) -> None:
    """Record successful publication in the local Nix database."""
    command = ["nix", "store", "sign", "--key-file", key_file, *paths]
    echo_cmd(command)
    subprocess.run(command, check=True)


def start_zstd_compressor(destination: IO) -> subprocess.Popen:
    return subprocess.Popen(
        ["zstd", "-zc", "-T0"],
        stdin=subprocess.PIPE,
        stdout=destination,
    )


def start_nar_packer(
    store_path: str,
    compressor_stdin: IO,
) -> subprocess.Popen:
    return subprocess.Popen(
        ["nix", "nar", "pack", store_path],
        stdout=compressor_stdin,
    )


def emit_csv(rows: list[tuple[str, str, str]], out: IO) -> None:
    """Write path, cache, and target rows."""
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(("path", "cache", "target"))
    writer.writerows(rows)
    out.flush()
    if out is not sys.stdout:
        out.close()


#
# Nix32 and signing
#

def nix32_encode(data: bytes) -> str:
    """Encode bytes using Nix's base32 alphabet."""
    length = (len(data) * 8 - 1) // 5 + 1
    result = ""
    for n in range(length - 1, -1, -1):
        i, j = divmod(n * 5, 8)
        c = data[i] >> j
        if i < len(data) - 1:
            c |= data[i + 1] << (8 - j)
        result += NIX32_ALPHABET[c & 0x1F]
    return result


def sri_sha256_to_nix32(sri: str) -> str:
    """Convert nix's `sha256-<b64>` SRI form to bare nix32."""
    assert sri.startswith("sha256-"), sri
    return nix32_encode(base64.b64decode(sri.removeprefix("sha256-")))


def load_signing_key(key_file: str) -> tuple[str, "nacl.signing.SigningKey"]:
    """Load a Nix secret key as a PyNaCl signing key."""
    line = Path(key_file).read_text().strip()
    name, b64 = line.split(":", 1)
    sk = base64.b64decode(b64)
    if len(sk) != 64:
        sys.exit(f"push.py: unexpected secret-key length {len(sk)} (want 64)")
    return name, nacl.signing.SigningKey(sk[:32])


def key_name(key_file: str) -> str:
    """Return the name from a Nix secret key."""
    return Path(key_file).read_text().split(":", 1)[0].strip()


#
# Closure resolution and filtering
#

def derivation_output_paths(drv: dict) -> set[str]:
    """Return all output store paths recorded by one derivation."""
    output_paths: set[str] = set()
    env = drv.get("env") or {}
    for output_name, output in (drv.get("outputs") or {}).items():
        path = output.get("path") or env.get(output_name)
        assert path, f"derivation output {output_name!r} has no store path"
        output_paths.add(
            path if path.startswith("/nix/store/") else "/nix/store/" + path
        )
    return output_paths


def closure_store_paths(packages: list[str]) -> tuple[list[str], dict, dict]:
    """Return (store paths, derivations, outputs by derivation).

    Store paths include every output and source in the transitive closure
    reported by `nix derivation show -r`.
    """
    drv_info = show_derivation_closure(packages)

    # `nix derivation show` has two JSON layouts:
    #
    #   v1: { "/nix/store/<name>.drv": { ... }, ... }
    #   v2: { "derivations": { "<name>.drv": { ... }, ... }, ... }
    #
    raw_drvs = drv_info.get("derivations", drv_info)
    drvs: dict = {Path(name).name: drv for name, drv in raw_drvs.items()}
    outputs_by_drv = {
        name: derivation_output_paths(drv)
        for name, drv in drvs.items()
    }

    src_paths: set[str] = set()
    for drv in drvs.values():
        sources = (
            list(drv.get("inputSrcs", []))
            + list((drv.get("inputs") or {}).get("srcs", []))
        )
        src_paths.update(
            src if src.startswith("/nix/store/") else "/nix/store/" + src
            for src in sources
        )

    output_paths = set().union(*outputs_by_drv.values())
    return sorted(output_paths | src_paths), drvs, outputs_by_drv


def path_info_entry(path_infos: dict, path: str) -> dict | None:
    """Return one path's metadata from either Nix JSON schema."""
    if "info" in path_infos:
        return path_infos["info"].get(Path(path).name)
    return path_infos.get(path)


def select_path_infos(path_infos: dict, paths: list[str]) -> dict:
    """Reduce batched path metadata to the selected paths."""
    if "info" in path_infos:
        return {
            "storeDir": path_infos["storeDir"],
            "info": {
                Path(path).name: path_info_entry(path_infos, path)
                for path in paths
            },
        }
    return {path: path_info_entry(path_infos, path) for path in paths}


def filter_local(paths: list[str], path_infos: dict) -> list[str]:
    """Keep paths registered in the active Nix store."""
    return [path for path in paths if path_info_entry(path_infos, path) is not None]


def filter_unsigned(paths: list[str], path_infos: dict) -> list[str]:
    """Keep paths with no signatures."""
    unsigned: list[str] = []
    for path in paths:
        info = path_info_entry(path_infos, path)
        assert info is not None
        if not info.get("signatures"):
            unsigned.append(path)
    return unsigned


def filter_not_upstream_signed(
    paths: list[str],
    path_infos: dict,
    our_keys: set[str],
) -> list[str]:
    """Keep unsigned paths and paths signed by one of our keys."""
    our_prefixes = tuple(f"{key}:" for key in our_keys)
    candidates: list[str] = []
    for path in paths:
        info = path_info_entry(path_infos, path)
        assert info is not None
        signatures = info.get("signatures") or []
        if not signatures or any(
            signature.startswith(our_prefixes)
            for signature in signatures
        ):
            candidates.append(path)
    return candidates


#
# Cache classification
#

def drv_cache_name(drv: dict) -> str | None:
    """Return a derivation's explicit structured cache selection."""
    env = drv.get("env") or {}
    structured_attrs = drv.get("structuredAttrs") or {}
    assert not env.get("useCache", ""), "useCache requires __structuredAttrs = true"

    cache_name = structured_attrs.get("useCache")
    assert cache_name is None or isinstance(cache_name, str)
    return cache_name


def classify_paths_by_cache(
    paths: list[str],
    drv_closure: dict,
    outputs_by_drv: dict,
    default_cache: str,
    cache_names: list[str],
) -> dict[str, list[str]]:
    """Assign paths to the default or an explicitly selected cache."""
    explicit_caches_by_path: dict[str, set[str]] = {}
    for drv_name, drv in drv_closure.items():
        cache_name = drv_cache_name(drv)
        if cache_name is None:
            continue
        if cache_name not in cache_names:
            sys.exit(
                f"push.py: {drv_name} selects undefined cache {cache_name!r}"
            )
        for path in outputs_by_drv[drv_name]:
            explicit_caches_by_path.setdefault(path, set()).add(cache_name)

    paths_by_cache: dict[str, list[str]] = {
        cache_name: []
        for cache_name in cache_names
    }
    for path in paths:
        explicit_caches = explicit_caches_by_path.get(path, set())
        if len(explicit_caches) > 1:
            choices = ", ".join(sorted(explicit_caches))
            sys.exit(
                f"push.py: {path} is assigned to conflicting caches: {choices}"
            )
        cache_name = next(iter(explicit_caches), default_cache)
        paths_by_cache[cache_name].append(path)
    return paths_by_cache


#
# Cache publication
#

def remote_narinfos(remote: str) -> set[str]:
    """List narinfo names at a remote cache root."""
    command = [
        "rsync",
        "--list-only",
        "--include=/*.narinfo",
        "--exclude=*",
        remote + "/",
    ]
    result = subprocess.run(
        command,
        capture_output=True, text=True, check=False,
    )
    names: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts and parts[-1].endswith(".narinfo"):
            names.add(parts[-1])
    return names


def write_build_ninja(
    scratch: Path,
    paths: list[str],
    key_file: str,
    path_infos: dict,
) -> None:
    """Write path metadata and a parallel packing Ninja file."""
    selected_path_infos = select_path_infos(path_infos, paths)
    (scratch / "path-info.json").write_text(json.dumps(selected_path_infos))
    lines = [
        "ninja_required_version = 1.5",
        "",
        "rule pack",
        f"  command = {SELF_PATH} pack-one {scratch} {key_file} $store_path",
        "  description = pack $store_path",
        "",
    ]
    for sp in paths:
        storehash = Path(sp).name[:32]
        lines += [
            f"build {storehash}.narinfo nar-raw/{storehash}.nar.zst: pack",
            f"  store_path = {sp}",
            "",
        ]
    (scratch / "build.ninja").write_text("\n".join(lines))


def regen_nar_symlinks(scratch: Path) -> None:
    """Rebuild content-hash NAR symlinks from generated narinfos."""
    nar_dir = scratch / "nar"
    if nar_dir.exists():
        shutil.rmtree(nar_dir)
    nar_dir.mkdir()
    for ni in scratch.glob("*.narinfo"):
        storehash = ni.stem
        url: str | None = None
        for line in ni.read_text().splitlines():
            if line.startswith("URL: "):
                url = line[len("URL: "):]
                break
        assert url is not None and url.startswith("nar/"), f"bad URL in {ni}"
        link = scratch / url
        # Identical compressed NARs share a URL, so one symlink is sufficient.
        if not link.is_symlink():
            link.symlink_to(Path("../nar-raw") / f"{storehash}.nar.zst")


def filter_remote_paths(remote: str, paths: list[str]) -> list[str]:
    """Keep paths whose narinfos are absent from this remote."""
    log(f"=== {len(paths)} candidate(s) for {remote} ===")
    if not paths:
        return paths
    log(f"  listing narinfos on {remote}...")
    have = remote_narinfos(remote)
    log(f"  remote has {len(have)} narinfo(s)")
    kept = [p for p in paths if (Path(p).name[:32] + ".narinfo") not in have]
    log(f"  {len(kept)} path(s) missing from remote")
    return kept


def publish_store_paths(
    remote: str,
    paths: list[str],
    key_file: str,
    path_infos: dict,
) -> None:
    """Build and publish a static cache, then mark its paths as published.

    The phases are:

    1. Write path metadata and a Ninja file in a temporary staging directory.
    2. Pack NARs and create signed narinfos in parallel.
    3. Build the content-hash NAR layout.
    4. Upload NARs first and narinfos second.
    5. Sign the local store paths only after both uploads succeed.
    """
    if not paths:
        log(f"  nothing to push to {remote}")
        return

    with tempfile.TemporaryDirectory(prefix="nix-cache-push.") as scratch_str:
        scratch = Path(scratch_str)
        log(f"  scratch dir: {scratch}")

        # Every static cache needs a nix-cache-info at its root; rsync picks
        # it up in the first push phase.
        (scratch / "nix-cache-info").write_text("StoreDir: /nix/store\n")

        log(f"  emitting build.ninja for {len(paths)} path(s)...")
        write_build_ninja(scratch, paths, key_file, path_infos)

        log("  running ninja (parallel pack + sign)...")
        run_ninja_build(scratch)

        log("  generating nar/ symlinks from narinfos...")
        regen_nar_symlinks(scratch)

        log(f"  pushing nars + nix-cache-info to {remote}...")
        rsync_cache_payload(scratch, remote)

        log(f"  pushing narinfos to {remote}...")
        rsync_cache_narinfos(scratch, remote)

    log("  recording signatures in local nix DB...")
    sign_local_store_paths(key_file, paths)
    log(f"  done with {remote}")


#
# Ninja worker
#

def pack_and_compress(store_path: str, dest_dir: Path) -> tuple[Path, str, int]:
    """Return a compressed NAR file, hash, and size."""
    fd, tmp_name = tempfile.mkstemp(dir=dest_dir)
    tmp_path = Path(tmp_name)
    with os.fdopen(fd, "wb+") as f:
        zstd = start_zstd_compressor(f)
        assert zstd.stdin is not None
        nar = start_nar_packer(store_path, zstd.stdin)
        zstd.stdin.close()
        nar_rc = nar.wait()
        zstd_rc = zstd.wait()
        if nar_rc != 0 or zstd_rc != 0:
            tmp_path.unlink(missing_ok=True)
            sys.exit(f"push.py pack-one: pack failed (nar={nar_rc}, zstd={zstd_rc})")
        f.seek(0)
        comphash = nix32_encode(file_digest(f, "sha256").digest())
        compsize = f.seek(0, os.SEEK_END)
    return tmp_path, comphash, compsize


def pack_one(scratch: Path, store_path: str, key_file: str) -> None:
    """Pack one path and write its signed narinfo."""
    basename = os.path.basename(store_path)
    storehash = basename[:32]

    nar_raw_dir = scratch / "nar-raw"
    nar_raw_dir.mkdir(exist_ok=True)

    store_info = json.loads((scratch / "path-info.json").read_text())
    if "info" in store_info:
        # Newer path-info JSON schema.
        info = store_info["info"][basename]
        store_dir = store_info["storeDir"]  # typically "/nix/store"
        references = info["references"]  # basenames
        deriver = info.get("deriver")    # basename or None
    else:
        # Legacy schema used by Nix 2.31 and earlier: full paths are
        # both the mapping keys and the reference/deriver values.
        info = store_info[store_path]
        store_dir = str(Path(store_path).parent)
        references = [Path(reference).name for reference in info["references"]]
        raw_deriver = info.get("deriver")
        deriver = Path(raw_deriver).name if raw_deriver else None

    nar_hash_nix32 = sri_sha256_to_nix32(info["narHash"])
    nar_size = info["narSize"]

    tmp_path, comphash, compsize = pack_and_compress(store_path, nar_raw_dir)
    tmp_path.rename(nar_raw_dir / f"{storehash}.nar.zst")

    name, signing_key = load_signing_key(key_file)
    # Nix fingerprints use full store paths for the path and references.
    full_refs = ",".join(f"{store_dir}/{r}" for r in references)
    fingerprint = f"1;{store_path};sha256:{nar_hash_nix32};{nar_size};{full_refs}"
    sig = base64.b64encode(signing_key.sign(fingerprint.encode()).signature).decode()

    lines = [
        f"StorePath: {store_path}",
        f"URL: nar/{comphash}.nar.zst",
        f"Compression: zstd",
        f"FileHash: sha256:{comphash}",
        f"FileSize: {compsize}",
        f"NarHash: sha256:{nar_hash_nix32}",
        f"NarSize: {nar_size}",
        f"References: {' '.join(references)}",
    ]
    if deriver:
        lines.append(f"Deriver: {deriver}")
    lines.append(f"Sig: {name}:{sig}")
    (scratch / f"{storehash}.narinfo").write_text("\n".join(lines) + "\n")


#
# Subcommands
#

CONFIG_DEFAULT = Path("nix-cache-ssh.yml")


def load_config(path: Path) -> dict:
    """Read and validate the named cache configuration."""
    if not path.exists():
        sys.exit(f"push.py: config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        sys.exit(f"push.py: {path} must contain a YAML mapping at the top level")

    default_cache = cfg.get("default")
    caches = cfg.get("caches")
    if not isinstance(default_cache, str) or not default_cache:
        sys.exit(f"push.py: {path} must define a non-empty 'default' cache name")
    if not isinstance(caches, dict) or not caches:
        sys.exit(f"push.py: {path} must define a non-empty 'caches' mapping")
    if default_cache not in caches:
        sys.exit(
            f"push.py: {path} default cache {default_cache!r} is not in 'caches'"
        )

    for cache_name, section in caches.items():
        if not isinstance(cache_name, str) or not cache_name:
            sys.exit(f"push.py: {path} contains an invalid cache name")
        if not isinstance(section, dict):
            sys.exit(f"push.py: {path} cache {cache_name!r} must be a mapping")
        for field in ("target", "secret_key_file"):
            value = section.get(field)
            if not isinstance(value, str) or not value:
                sys.exit(
                    f"push.py: {path} cache {cache_name!r} missing {field!r}"
                )
    return cfg


def add_input_derivations(packages: list[str]) -> list[str]:
    """Add each available inputDerivation installable."""
    installables: list[str] = []
    for package in packages:
        package_without_outputs = package.split("^", 1)[0]
        installables.append(package)
        if package_has_input_derivation(package_without_outputs):
            installables.append(package_without_outputs + ".inputDerivation")
        else:
            log(f"{package_without_outputs} has no inputDerivation; skipping it")
    return installables


def cmd_push(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    default_cache: str = cfg["default"]
    caches: dict = cfg["caches"]
    cache_names = list(caches)
    cache_targets = {
        cache_name: cache["target"].rstrip("/")
        for cache_name, cache in caches.items()
    }
    cache_keys = {
        cache_name: cache["secret_key_file"]
        for cache_name, cache in caches.items()
    }
    our_keys = {
        key_name(key_file)
        for key_file in cache_keys.values()
    }
    requested_packages: list[str] = args.packages
    packages = (
        requested_packages
        if args.no_input_derivation
        else add_input_derivations(requested_packages)
    )

    mode = (
        "--check-remote (list remote, push what's missing)"
        if args.check_remote
        else "default (skip locally-signed)"
    )
    log(f"Mode: {mode}")
    log(f"Config:          {args.config}")
    log(f"Default cache:   {default_cache}")
    log(f"Our key name(s): {', '.join(sorted(our_keys))}")
    for cache_name in cache_names:
        log(
            f"Cache {cache_name}: {cache_targets[cache_name]}"
            f"  (key: {cache_keys[cache_name]})"
        )
    log(
        f"Packages ({len(requested_packages)}): "
        + " ".join(requested_packages)
    )
    log(f"Installables ({len(packages)}): {' '.join(packages)}")

    log("Resolving derivation closure...")
    all_paths, drv_closure, outputs_by_drv = closure_store_paths(packages)

    log(f"Closure has {len(all_paths)} store path(s), {len(drv_closure)} drv(s)")

    log("Querying local path metadata...")
    path_infos = query_path_infos(all_paths)
    local_paths = filter_local(all_paths, path_infos)
    dropped = len(all_paths) - len(local_paths)
    log(f"{len(local_paths)} path(s) present locally (dropped {dropped})")

    if args.check_remote:
        log("Filtering out upstream-signed paths (keeping unsigned and our-keys-only)...")
        candidates = filter_not_upstream_signed(
            local_paths,
            path_infos,
            our_keys,
        )
    else:
        log("Filtering to unsigned (locally-built, never-pushed) paths...")
        candidates = filter_unsigned(local_paths, path_infos)
    log(f"{len(candidates)} path(s) candidate for push")

    if not candidates:
        log("Nothing to push. Done.")
        emit_csv([], args.output)
        return

    log("Classifying paths by cache...")
    paths_by_cache = classify_paths_by_cache(
        candidates,
        drv_closure,
        outputs_by_drv,
        default_cache,
        cache_names,
    )
    split = ", ".join(
        f"{cache_name}={len(paths_by_cache[cache_name])}"
        for cache_name in cache_names
    )
    log(f"Split: {split}")

    if args.check_remote:
        paths_to_push = {
            cache_name: filter_remote_paths(
                cache_targets[cache_name],
                paths_by_cache[cache_name],
            )
            for cache_name in cache_names
        }
    else:
        paths_to_push = paths_by_cache

    csv_rows = [
        (path, cache_name, cache_targets[cache_name])
        for cache_name in cache_names
        for path in paths_to_push[cache_name]
    ]
    emit_csv(csv_rows, args.output)

    if args.dry_run:
        log(f"Dry-run: {len(csv_rows)} path(s) would be pushed. Not pushing.")
        return

    for cache_name in cache_names:
        publish_store_paths(
            cache_targets[cache_name],
            paths_to_push[cache_name],
            cache_keys[cache_name],
            path_infos,
        )
    log("All done.")


def cmd_pack_one(args: argparse.Namespace) -> None:
    pack_one(Path(args.scratch_dir), args.store_path, args.key_file)


#
# Argument parsing
#

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="nix-cache-push",
        description="Push Nix closures to configured static caches.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    push = sub.add_parser("push", help="orchestrate a push (top-level entry)")
    push.add_argument(
        "-c",
        "--config",
        type=Path,
        default=CONFIG_DEFAULT,
        help=f"YAML config file (default: ./{CONFIG_DEFAULT})",
    )
    push.add_argument(
        "--check-remote",
        action="store_true",
        help="list remote narinfos and push only what's missing",
    )
    push.add_argument(
        "--no-input-derivation",
        action="store_true",
        help="do not detect and add package inputDerivations",
    )
    push.add_argument(
        "--dry-run",
        action="store_true",
        help="print CSV of what would be pushed; don't pack/sign/rsync",
    )
    push.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="write CSV of push set to PATH (default: stdout; `-` also = stdout)",
    )
    push.add_argument("packages", nargs="+", help="Nix flake package references")
    push.set_defaults(func=cmd_push)

    # Positional order below matches how build.ninja invokes us (see
    # write_build_ninja): SCRATCH KEY_FILE STORE_PATH.
    pack = sub.add_parser(
        "pack-one",
        help="(internal) pack one store path; invoked by ninja",
    )
    pack.add_argument("scratch_dir")
    pack.add_argument("key_file")
    pack.add_argument("store_path")
    pack.set_defaults(func=cmd_pack_one)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
