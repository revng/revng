{ pkgs, stdenv, lib, fetchPrivateUrl }:
let
  # ----------------------------------------------------------------
  # Windows rootfs derivations. revng's api-set-schema test does:
  #   find $INSTALL_ROOT/share/roots/windows/<name> \
  #     -iname apisetschema.dll -path "*/System32/*"
  # then runs `revng internal dump-api-set-schema` against the DLL.
  # We mirror orchestra/.orchestra/config/components/rootfs.yml:
  #   * docker pull mcr.microsoft.com/windows/servercore (servercore
  #     images carry apisetschema.dll for x86-64 and aarch64),
  #   * Microsoft eval-ISO downloads (Win7, Win8, Win8.1) carrying
  #     install.wim with a Windows install image.
  # In each case the rootfs is trimmed to just apisetschema.dll —
  # everything else is dropped, so the install size stays tiny.
  # ----------------------------------------------------------------

  # Common installPhase. Lift the keep-only-apisetschema.dll filter
  # out so both helpers share the same trimming logic.
  trimAndInstall = name: ''
    # Keep only apisetschema.dll regular files. Drop everything else,
    # including the many Windows-style symlinks that point at paths
    # with literal "C:/..." segments (those would otherwise fail
    # nixpkgs' noBrokenSymlinks check).
    find rootfs -not -type d ! \( -type f -iname 'apisetschema.dll' \) -delete
    chmod -R u+rwX rootfs/
    find rootfs -type d -empty -delete

    mkdir -p "$out/share/roots/windows/${name}"
    cp -a rootfs/. "$out/share/roots/windows/${name}/"
    chmod -R u+rwX "$out/share/roots/windows/${name}/"
  '';

  # Docker-image variant. skopeo+umoci pull the image (network is
  # allowed inside the FOD), umoci raw unpack lays it out as a tree.
  mkWinContainerRootfs =
    { name, imageUrl, outputHash }:
    stdenv.mkDerivation {
      name = "rootfs-${name}";
      __structuredAttrs = true;
      # apisetschema.dll is a Microsoft binary; keep it off the public cache.
      useCache = "private";
      outputHashAlgo = "sha256";
      outputHashMode = "recursive";
      inherit outputHash;
      unpackPhase = "true";
      nativeBuildInputs = with pkgs; [ skopeo umoci cacert ];
      buildPhase = ''
        export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
        # No /etc/containers/policy.json in the nix sandbox; accept
        # the registry image at its content-addressed digest instead.
        skopeo --insecure-policy --override-os windows copy \
          "${imageUrl}" \
          oci:rootfs-oci:latest
        umoci raw unpack --rootless --image rootfs-oci rootfs
        rm -rf rootfs-oci
        find rootfs -maxdepth 3 -type d -exec chmod 755 {} \;
      '';
      installPhase = trimAndInstall name;
    };

  # WIM/ISO variant. We fetch the Microsoft eval ISO via fetchurl
  # (itself a FOD, hash-pinned) so the outer derivation can stay
  # network-free, then peel install.wim out with 7z and unpack the
  # rootfs from the wim image.
  mkWinWimRootfs =
    { name, isoUrl, isoHash, wimPath, outputHash }:
    let
      iso = fetchPrivateUrl {
        url = isoUrl;
        sha256 = isoHash;
      };
    in
    stdenv.mkDerivation {
      name = "rootfs-${name}";
      __structuredAttrs = true;
      # apisetschema.dll is a Microsoft binary; keep it off the public cache.
      useCache = "private";
      outputHashAlgo = "sha256";
      outputHashMode = "recursive";
      inherit outputHash;
      unpackPhase = "true";
      nativeBuildInputs = with pkgs; [ p7zip ];
      buildPhase = ''
        # ISO → sources/install.wim into cwd.
        7z x -y "${iso}" "${wimPath}"
        # install.wim → rootfs tree. The wim uses a "1/" image index
        # prefix for single-image archives — flatten it.
        7z x -y "${wimPath}" -orootfs
        if [ -d rootfs/1 ]; then
          mv rootfs/1/* rootfs/ && rmdir rootfs/1
        fi
        rm -rf sources/
      '';
      installPhase = trimAndInstall name;
    };
in
{
  "rootfs/windows-x86-64" = mkWinContainerRootfs {
    name = "windows-x86-64";
    imageUrl = "docker://mcr.microsoft.com/windows/servercore@sha256:a4d6cb8427e90fd15e39bef29e0e800465989793d2a0826d3b7ceba78af0fe34";
    outputHash = "sha256-Q8CEHQ4qaOszm3p59dUTOLDZ6fQIyAerXOtJvXsG5SA=";
  };

  "rootfs/windows-aarch64" = mkWinContainerRootfs {
    name = "windows-aarch64";
    imageUrl = "docker://mcr.microsoft.com/windows/servercore@sha256:6e508f22063e2ab597c32909b66e3d4dff8914e385bd18410a4d7331d1b90dd4";
    outputHash = "sha256-uF3Td4+7y+5vzRzF7/4pI7LoF4aqAkX0BxxE9BV1joE=";
  };

  "rootfs/windows-7-x86" = mkWinWimRootfs {
    name = "windows-7-x86";
    isoUrl = "http://care.dlservice.microsoft.com/dl/download/evalx/win7/x86/EN/7600.16385.090713-1255_x86fre_enterprise_en-us_EVAL_Eval_Enterprise-GRMCENEVAL_EN_DVD.iso";
    isoHash = "sha256-JAEIedmOkKmJxCClA7B9nXhO8z+zxnkgwZeZYdbNe1c=";
    wimPath = "sources/install.wim";
    outputHash = "sha256-W1MEu4HDtmd88hilkElEzJmHOntl66o4x6Y4AjoGYGQ=";
  };

  "rootfs/windows-8-x86-64" = mkWinWimRootfs {
    name = "windows-8-x86-64";
    isoUrl = "https://download.microsoft.com/download/5/3/C/53C31ED0-886C-4F81-9A38-F58CE4CE71E8/9200.16384.WIN8_RTM.120725-1247_X64FRE_ENTERPRISE_EVAL_EN-US-HRM_CENA_X64FREE_EN-US_DV5.ISO";
    isoHash = "sha256-kV250X/RO5KLg4JeNFP0U1zJrf1zIjAcMPxmsqwPqcw=";
    wimPath = "sources/install.wim";
    outputHash = "sha256-XrIxX8XoWdbBalMJMLA5PH2MW8cKLT4ymDNvgl1dCr4=";
  };

  "rootfs/windows-8-1-x86-64" = mkWinWimRootfs {
    name = "windows-8-1-x86-64";
    isoUrl = "https://download.microsoft.com/download/B/9/9/B999286E-0A47-406D-8B3D-5B5AD7373A4A/9600.16384.WINBLUE_RTM.130821-1623_X64FRE_ENTERPRISE_EVAL_EN-US-IRM_CENA_X64FREE_EN-US_DV5.ISO";
    isoHash = "sha256-MwwWWoczzw7sb5CUqPbdOijeR4Er/6YNcnmpqdHdGmE=";
    wimPath = "sources/install.wim";
    outputHash = "sha256-Y5rwBepI4skTSaQCrlf8FLFeKG9WV86kTHB9CsBkQbk=";
  };
}
