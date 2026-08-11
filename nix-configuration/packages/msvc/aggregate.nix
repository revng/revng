{ pkgs, stdenv, nixpkgsLib }:
{ name, vsmanUrl, vsmanSha256, selected, outputHash }:
let
  vsmanSha256Hex = builtins.convertHash {
    hash = vsmanSha256;
    hashAlgo = "sha256";
    toHashFormat = "base16";
  };
in
stdenv.mkDerivation {
  inherit name outputHash;
  __structuredAttrs = true;
  useCache = "private";
  outputHashMode = "recursive";
  outputHashAlgo = "sha256";
  nativeBuildInputs = [ pkgs.curl pkgs.jq pkgs.nix pkgs.cacert ];
  dontUnpack = true;
  buildPhase = ''
    export XDG_CACHE_HOME="$TMPDIR/nix-cache"
    export NIX_STATE_DIR="$TMPDIR/nix-state"
    export NIX_CONFIG="experimental-features ="
    export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"

    curl -fsSL -o vsman "${vsmanUrl}"
    echo "${vsmanSha256Hex}  vsman" | sha256sum -c

    cat > selected.json <<'EOF'
    ${builtins.toJSON selected}
    EOF

    cat > wrapper.nix <<EOF
    let
      lib        = import ${nixpkgsLib}/lib;
      parseVsman = import ${./parse-vsman.nix};
      vsmanText  = builtins.readFile ./vsman;
      selected   = builtins.fromJSON (builtins.readFile ./selected.json);
    in parseVsman lib vsmanText selected
    EOF

    nix-instantiate --eval --strict --json wrapper.nix > triples.json

    jq -c '.[]' triples.json | while read -r row; do
      component=$(jq -r .component <<<"$row")
      filename=$( jq -r .filename  <<<"$row")
      url=$(      jq -r .url       <<<"$row")
      sha=$(      jq -r .sha256    <<<"$row")
      dest="$out/$component/$filename"
      install -D /dev/null "$dest"
      curl -fsSL -o "$dest" "$url"
      echo "$sha  $dest" | sha256sum -c
    done
  '';
  installPhase = "true";
}
