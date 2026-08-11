{ stdenv, revngPackages }:
# Public aggregate of the `apisetschema.dll` files pulled out of each
# private-cache `rootfs/windows-*` derivation. The Windows rootfses
# themselves are non-redistributable (`useCache = "private"`), but the
# assembled DLL-only tree here contains just Microsoft-signed binary
# blobs with no textual references back to their source store paths, so
# the aggregate has zero runtime store refs and can safely ride the
# public cache — letting users pull the `apisetschema.dll`s that
# `test/revng` needs without any private-cache access.
stdenv.mkDerivation {
  name = "apisetschema-dlls";
  dontUnpack = true;
  installPhase = ''
    mkdir -p "$out/share/roots/windows"
    cp -r ${revngPackages."rootfs/windows-x86-64"}/share/roots/windows/.     "$out/share/roots/windows/"
    cp -r ${revngPackages."rootfs/windows-aarch64"}/share/roots/windows/.    "$out/share/roots/windows/"
    cp -r ${revngPackages."rootfs/windows-7-x86"}/share/roots/windows/.      "$out/share/roots/windows/"
    cp -r ${revngPackages."rootfs/windows-8-x86-64"}/share/roots/windows/.   "$out/share/roots/windows/"
    cp -r ${revngPackages."rootfs/windows-8-1-x86-64"}/share/roots/windows/. "$out/share/roots/windows/"
  '';
}
