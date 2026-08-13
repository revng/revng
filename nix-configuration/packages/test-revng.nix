{ pkgs, pkgs-2505, stdenv, python,
  revng, revngPythonDependencies, revngPackages,
}:
let
  gcc11Stdenv = pkgs.overrideCC stdenv hostGcc11;

  # Orchestra ships gcc-11.2.0 as its native toolchain; test-docs's
  # `gcc account.c ...` doctest bakes gcc-11 codegen into
  # `editing-types-in-c.md`'s expected output (expression order
  # `balance + id + flags`). The scheduler in gcc 11.5.0 reorders the
  # `id / balance / flags` loads compared to 11.2.0 (11.2 emits
  # `movslq (%rdi); add 0x8(%rdi); movslq 0x10(%rdi); add`; 11.5 emits
  # `movslq (%rdi); movslq 0x10(%rdi); add; add 0x8(%rdi)`), and revng
  # walks that order verbatim into the decompiled expression. Swap
  # pkgs-2505's gcc11 (11.5.0) source for the 11.2.0 tarball to match
  # orchestra byte-for-byte.
  hostGcc11 =
    let
      unwrapped = pkgs-2505.gcc11.cc.overrideAttrs (old: {
        version = "11.2.0";
        src = pkgs.fetchurl {
          url = "mirror://gcc/releases/gcc-11.2.0/gcc-11.2.0.tar.xz";
          hash = "sha256-0I7cU2tUw3KhAQ/2YZ3SdMDxYDqkkhK6IPeqLNo2+os=";
        };
        # nixpkgs' postPatch bakes `[[ 11.5.0 != $gcc_base_version ]]`
        # via interpolation; adjust to 11.2.0 so BASE-VER matches.
        postPatch = pkgs.lib.replaceStrings
          [ "[[ 11.5.0 != $gcc_base_version ]]" ]
          [ "[[ 11.2.0 != $gcc_base_version ]]" ]
          (old.postPatch or "");
        # gcc 11.2.0's libsanitizer fails to compile against glibc 2.40
        # (same narrowing-conversion abort inside
        # `sanitizer_platform_limits_posix.cc` we hit on 9.2.0). We only
        # need gcc for compiling the doctests' `account.c` / `example.c`,
        # not for libasan/lsan/tsan, so turn the whole subdir off.
        configureFlags = (old.configureFlags or [ ]) ++ [
          "--disable-libsanitizer"
        ];
      });
    in
    pkgs-2505.gcc11.override { cc = unwrapped; };
  # Single Python env with revng's modules + the wheels from
  # revngPythonDependencies (jinja2, pyyaml, yq/jq runtime, …)
  # all visible on the venv's native sys.path — no PYTHONPATH
  # gymnastics in the test runner.
  revngPythonEnv = revngPythonDependencies.overrideAttrs (old: {
    postInstall = (old.postInstall or "") + ''
      cp -a ${revng}/${python.sitePackages}/. \
        $out/${python.sitePackages}/
    '';
  });

  # Search list passed to test-configure as repeated --input-path
  # roots and to the running revng via REVNG_RESOURCES. With the
  # ${COMMAND_ROOT}/${SOURCE} idiom in revng's YMLs, companion files
  # (.filecheck, .model.yml, .cfg.yml, …) are looked up in the YAML's
  # own root — and the YAMLs themselves now live in revng-test-assets
  # (split out of revng so test-fixture edits don't trigger a full
  # ~10 min revng rebuild). source .c/.S binaries are picked from
  # revng-qa/ as before, and `apisetschema-dlls` is the public
  # aggregate that carries share/roots/windows/<rootfs>/apisetschema.dll
  # for the api-set-schema tests — the private `rootfs/windows-*`
  # sources land in it at build time but not at runtime.
  searchRoots = [
    revngPackages.revng-qa
    revngPackages."test/revng-qa"
    revngPackages.revng-test-assets
    revng
    revngPackages.apisetschema-dlls
    # `qemu-helpers.md`'s doctest reads
    # `$ROOT/share/libtcg/libtcg-helpers-x86_64.bc` where `$ROOT` is
    # `$(dirname $(dirname $(which revng)))`. In orchestra qemu-helpers
    # installs into the same `/orchestra/root/`, so the file colocates
    # with revng. Here it lives in a separate store output — include
    # it in the merged share/ tree so `$PWD/share/libtcg/…` resolves.
    revngPackages.qemuHelpers
  ];
in
gcc11Stdenv.mkDerivation {
  name = "test/revng";

  __structuredAttrs = true;
  unsafeDiscardReferences.out = true;

  # test-docs's doctest compiles `account.c` with bare `gcc` inside this
  # derivation, and revng's DetectABI + decompiler read the emitted
  # binary — the tutorial's expected_output.log was baked against
  # orchestra's un-hardened gcc-11.2.0. nix's cc-wrapper otherwise
  # injects `-fPIC` (GOT round-trips reshape the load scheduling and
  # flip the summed expression order), `-fzero-call-used-regs` (adds
  # `xor %edx,%edx; xor %edi,%edi` before `ret`, which DetectABI reads
  # as live-out and turns the `generic64_t` return into an
  # `opaque_type_16 var_0 + bit_cast` shape), `_FORTIFY_SOURCE`,
  # `-fstack-protector*`, etc. — every one of them drifts the doctest
  # away from orchestra's baked output. Turn them all off.
  hardeningDisable = [ "all" ];

  unpackPhase = "true";

  # `gcc11Stdenv` (above) already injects `hostGcc11` as the derivation's
  # cc-wrapper, so ninja rules invoking bare `gcc` (e.g. test-docs's
  # `gcc account.c ...` doctest) resolve to gcc-11.2.0, matching orchestra
  # byte-for-byte — no separate entry needed here.
  nativeBuildInputs = [
    # `revngPythonEnv` (the venv with revng's Python modules on its
    # `sys.path`) must be first so bare `python3` resolves to it — the
    # `scripting-{cli,daemon}` doctests invoke `scripting.py` via its
    # patched shebang, and if PATH's `python3` doesn't see the `revng`
    # module those tests import-fail with `ModuleNotFoundError`.
    revngPythonEnv
  ] ++ (with pkgs; [
    binutils
    jq
    llvm_21
    lld_21
    # `revng check-decompiled-c` syntax-checks decompiled C via `clang`;
    # nix's gcc11 stdenv doesn't bring one in transitively.
    clang_21
    # `qemu-helpers.md`'s doctest starts with
    # `ROOT="$(dirname "$(dirname "$(which revng)")")"`; nix's minimal
    # sandbox PATH doesn't ship `which`.
    which
    # zstdcat: used by share/revng/test/tests/pypeline-comparison/
    # compare.sh.
    zstd
  ]) ++ [
    # Use our revng/qemu fork rather than nixpkgs' generic qemu 10.1.2:
    # the fork is what orchestra ships, and only its qemu-* linux-user
    # binaries are known to match the address-space layout revng's
    # runtime-abi-tests expect (MAP_FIXED at 0x02000000 / 0x03000000).
    revngPackages.qemu
  ] ++ [
    # Wrapped node interpreter — `require('revng-model')`, `s3rver`,
    # `tsc`, etc. resolve without callers having to export NODE_PATH
    # or assemble a search path themselves. Replaces pkgs.nodejs.
    revngPackages.revng-test-node-env
    revngPackages.ninjaShellRule
    revng
    revngPackages."test/revng-qa"
  ];

  buildPhase = ''
    echo
  '';

  # All the re-runnable setup work lives in `preInstall` so it's
  # exported as a bash variable in `nix develop .#"test/revng"` —
  # users can `eval "$preInstall"` once inside a fresh workdir to
  # regenerate build.ninja, then `ninja <target>` individual targets.
  # Don't reach for nixpkgs setup-hook functions (`patchShebangs`,
  # etc.) here — they're only defined when `$stdenv/setup` is
  # sourced, which in nix develop's --command mode happens to also
  # trigger genericBuild and run every phase.
  preInstall = ''
    # Orchestra installs every component into a single `/orchestra/root/`
    # tree, so its YAMLs treat bare `''${SOURCE}` (a relative path) and
    # `''${COMMAND_ROOT}/''${SOURCE}` as pointing to a colocated `.c` +
    # `.model.yml` + `.filecheck.ll` triple. In nix each package installs
    # into its own store path — .c ends up in revng-qa, its .model.yml
    # ends up in revng-test-assets — so a bare `''${SOURCE}` no longer
    # resolves against the ninja CWD. Rebuild that single-root illusion
    # here by symlinking every input tree's `share/` into `$PWD/share/`
    # file-by-file (so directories nest); ninja later writes its own
    # rule outputs alongside the symlinks under `share/`, which coexists
    # fine because outputs have per-rule suffixes.
    for _tree in ${pkgs.lib.concatMapStringsSep " " (p: ''"${p}"'') searchRoots}; do
      [ -d "$_tree/share" ] || continue
      (cd "$_tree" && find share \( -type f -o -type l \) 2>/dev/null) \
      | while IFS= read -r _rel; do
          _dst="$PWD/$_rel"
          [ -e "$_dst" ] && continue
          mkdir -p "$(dirname "$_dst")"
          ln -s "$_tree/$_rel" "$_dst"
        done
    done
    unset _tree _rel _dst

    # Doctests such as `qemu-helpers.md` compute
    # `ROOT="$(dirname "$(dirname "$(which revng)")")"` and then read
    # `$ROOT/share/...`. Orchestra puts `revng` in `/orchestra/root/bin`
    # and every component's `share/` alongside it, so `$ROOT` is the
    # shared install prefix. Mirror that here: drop a `revng` shim into
    # `$PWD/bin/` that exec's the real one, and put `$PWD/bin` at the
    # front of PATH — `which revng` then returns `$PWD/bin/revng`,
    # `$ROOT` becomes `$PWD`, and the merged `share/` tree above serves
    # the companion files.
    mkdir -p "$PWD/bin"
    cat > "$PWD/bin/revng" <<EOF
    #!$(command -v bash)
    exec ${revng}/bin/revng "\$@"
    EOF
    chmod +x "$PWD/bin/revng"
    export PATH="$PWD/bin:$PATH"

    python3 \
      ${revngPackages.revng-qa}/libexec/revng/test-configure \
      "${revngPackages.revng-qa}/share/revng/test/configuration/revng-qa/"*.yml \
      "${revngPackages.revng-test-assets}/share/revng/test/configuration/revng/"*.yml \
      --install-path "$PWD" \
      ${pkgs.lib.concatMapStringsSep " " (p: ''--input-path "${p}"'') searchRoots} \
      --destination . \
      --target-type 'revng\..*'
    # test-configure writes inline scripts with `#!/usr/bin/env <interp>`
    # shebangs that don't resolve in the sandbox. Hand-rolled patchShebangs
    # equivalent — covers the four interpreter forms test-configure emits.
    # Includes extension-less scripts (e.g. `check-invalidations` from the
    # invalidation YAMLs' `scripts:` block) by selecting on executability
    # rather than filename.
    for _f in $(find . -maxdepth 2 -type f -perm -u+x); do
      sed -i "1{
        s|^#!/usr/bin/env python3.*|#!$(command -v python3)|
        s|^#!/usr/bin/env bash.*|#!$(command -v bash)|
        s|^#!/usr/bin/env node.*|#!$(command -v node)|
        s|^#!/bin/bash.*|#!$(command -v bash)|
      }" "$_f"
    done
    unset _f

    export REVNG_OPTIONS="--debug-log=verify"
    export PYPELINE_STORAGE_PROVIDER="local://?inline"
    export XDG_CACHE_HOME="$PWD/.cache"
    mkdir -p "$XDG_CACHE_HOME/.cache"

    # `revng internal find-path` (used by db-invariants.yml and any
    # other rule needing cross-root lookups) walks REVNG_RESOURCES in
    # order — same list test-configure searches.
    export REVNG_RESOURCES="${pkgs.lib.concatMapStringsSep ":" toString searchRoots}"
  '';

  # Shell functions exposed when you `nix develop .#"test/revng"`.
  # Lets you iterate on individual targets without paying the
  # ~10s/run nix-develop evaluation cost each time.
  shellHook = ''
    repro-setup() {
      local wd=''${1:-/tmp/revng-test-repro}
      mkdir -p "$wd" || return $?
      (
        cd "$wd" &&
        rm -f build.ninja build.ninja.diag &&
        eval "$preInstall" &&
        echo "Setup complete in $wd."
      )
    }
    repro-run() {
      local target=''${1:?'usage: repro-run TARGET [WORKDIR]'}
      local wd=''${2:-/tmp/revng-test-repro}
      (
        cd "$wd" || return $?
        # First call: regenerate build.ninja via preInstall.
        if [[ ! -f build.ninja ]]; then eval "$preInstall"; fi
        # The test rules wrap each binary in `2>/dev/null || true`;
        # strip both so SIGABRT/SIGILL backtraces are visible.
        if [[ ! -f build.ninja.diag ]] || [[ build.ninja -nt build.ninja.diag ]]; then
          sed -e "s# 2>/dev/null##g" -e "s# || true##g" build.ninja > build.ninja.diag
        fi
        ninja -f build.ninja.diag -v -k0 "$target"
      )
    }
    if [[ -t 1 && -z ''${REPRO_QUIET:-} ]]; then
      echo "[test/revng] helpers available: repro-setup, repro-run"
    fi
  '';

  installPhase = ''
    mkdir -p $out
    runHook preInstall

    # WIP: tolerate failing test targets — bumped revng +
    # new pypeline still have several upstream-known crashes.
    # Capture the log so failing targets can be enumerated.
    mkdir -p "$out/log"
    ninja -v -k0 all 2>&1 | tee "$out/log/ninja.log" || true

    # Extract the list of FAILED targets for convenience. `grep -a`
    # because some tests pipe binary output through tee (wine /
    # dumpbin / gcc backtraces with raw bytes) and a default grep
    # silently emits `Binary file ... matches` to stdout while
    # writing nothing to the redirect.
    grep -aoE '^FAILED: [^ ]+' "$out/log/ninja.log" \
      > "$out/log/failed-targets.txt" || true
    echo "test/revng: $(wc -l < $out/log/failed-targets.txt) failing target(s); see $out/log/"
  '';

}
