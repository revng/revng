{ pkgs, python, inputs, nanobind }:
let
  workspace = inputs.uv2nix.lib.workspace.loadWorkspace {
    workspaceRoot = pkgs.lib.cleanSourceWith {
      name = "revng-python-dependencies";
      src = ./.;
      filter = path: type: baseNameOf path != "default.nix";
    };
  };
  pythonBase = pkgs.callPackage inputs.pyproject-nix.build.packages {
    inherit python;
  };
  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };
  pythonSet = pythonBase.overrideScope (
    pkgs.lib.composeManyExtensions [
      inputs.pyproject-build-systems.overlays.wheel
      overlay
      (
        final: prev:
        let
          addSetuptools = drv: drv.overrideAttrs (old: {
            nativeBuildInputs =
              (old.nativeBuildInputs or [ ]) ++ final.resolveBuildSystem { setuptools = [ ]; };
          });
        in
        {
          grandiso = addSetuptools prev.grandiso;
          jsonschema = addSetuptools prev.jsonschema;
          vivisect-vstruct-wb = addSetuptools prev.vivisect-vstruct-wb;
          # Drop python-idb's 305 MB tests/ fixture tree from its `src`
          # so it doesn't ride the runtime closure of every consumer.
          python-idb = (addSetuptools prev.python-idb).overrideAttrs (old: {
            src = pkgs.runCommand "python-idb-src-trimmed" { } ''
              cp -r ${old.src} $out
              chmod -R u+w $out
              rm -rf $out/tests
            '';
          });
          llvmcpy = addSetuptools prev.llvmcpy;
          # psycopg-c needs pg_config + libpq headers on top of setuptools.
          psycopg-c = (addSetuptools prev.psycopg-c).overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
              pkgs.libpq.pg_config
              pkgs.libpq
            ];
          });
          # hexdump's sdist unpacks into `data/` with no pyproject.toml at
          # the top; cd into the actual project dir before building.
          hexdump = (addSetuptools prev.hexdump).overrideAttrs (old: {
            preBuild = (old.preBuild or "") + ''
              cd /build
              buildDir=$(find . -maxdepth 1 -type d \
                -exec test -e "{}/pyproject.toml" -o -e "{}/setup.py" \; \
                -print -quit)
              cd "$buildDir"
            '';
          });
        }
      )
    ]
  );
  venv = pythonSet.mkVirtualEnv "revng-python-dependencies" workspace.deps.default;
in
# Drop nanobind into the venv's site-packages so it's discoverable
# alongside the uv2nix-resolved wheels — callers can use this attr
# as a single Python env (no PYTHONPATH gymnastics).
venv.overrideAttrs (old: {
  postInstall = (old.postInstall or "") + ''
    cp -a ${nanobind}/${python.sitePackages}/nanobind \
      $out/${python.sitePackages}/nanobind
  '';
})
