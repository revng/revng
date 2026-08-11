# Walk a VS17 `.vsman` and return the transitive list of payloads
# required by the given root component IDs, as
# `[ { component; filename; url; sha256 } ]` — filename slashes
# normalised to POSIX and sha256 kept as the vsman's hex string.
#
# Takes `lib` up front so the same file can be evaluated both from
# inside nixpkgs (mode a: `pkgs.lib`) and from a bare
# `nix-instantiate --eval` (mode b: `import ${nixpkgs-lib}/lib`).
lib: vsmanText: selected:
let
  packages = (builtins.fromJSON vsmanText).packages;

  englishOnly = builtins.filter
    (p: !(p ? language) || lib.toLower p.language == "en-us")
    packages;

  byLowerId = lib.listToAttrs
    (map (p: { name = lib.toLower p.id; value = p; }) englishOnly);

  find = id:
    if byLowerId ? ${id} then [ byLowerId.${id} ]
    else builtins.trace "Could not find ${id}" [ ];

  selectedLower = map lib.toLower selected;
  roots = builtins.filter (p: builtins.elem (lib.toLower p.id) selectedLower) englishOnly;

  depsOf = pkg:
    if pkg ? dependencies
    then map lib.toLower (builtins.attrNames pkg.dependencies)
    else [ ];

  walk = pkg: stack:
    if builtins.elem pkg.id stack then [ ]
    else [ (lib.toLower pkg.id) ]
         ++ lib.flatten (map (depId:
              map (p: walk p (stack ++ [ pkg.id ])) (find depId)
            ) (depsOf pkg));

  needed = lib.unique (lib.flatten (map (r: walk r [ ]) roots));

  payloadsOf = pkgId:
    lib.flatten (map (pkg:
      if pkg ? payloads
      then map (payload: {
        component = pkg.id;
        filename = builtins.replaceStrings [ "\\" ] [ "/" ] payload.fileName;
        sha256 = payload.sha256;
        url = payload.url;
      }) pkg.payloads
      else [ ]
    ) (find pkgId));
in
lib.flatten (map payloadsOf needed)
