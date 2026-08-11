# WIP: the host has to be x86-64. VS17 toolchains support AArch64 hosts as well.

{
  pkgs,
  stdenv,
  fetchPrivateUrl,
  inputs,
}:
let
  lib = pkgs.lib;
  vs17Payloads = import ./aggregate.nix { inherit pkgs stdenv; nixpkgsLib = inputs.nixpkgs-lib; } {
    name = "vs17-payloads";
    selected = [
      "Microsoft.VisualStudio.Component.VC.14.37.17.7.x86.x64"
      "Microsoft.VisualStudio.Component.VC.14.37.17.7.ARM64"
      "Microsoft.VisualStudio.Component.Windows11SDK.22621"
    ];
    # $ URL=$(curl -L https://aka.ms/vs/17/release/channel | jq -r '.channelItems | .[] | select(.id | contains("Microsoft.VisualStudio.Manifests.VisualStudio")).payloads[0].url')
    vsmanUrl = "https://download.visualstudio.microsoft.com/download/pr/5011a9cc-e8ef-42cb-ad72-87de1031accc/b674dd46f83a684142689b3a86c73ba5f2e2477018087e4df218516214bbad58/VisualStudio.vsman";
    vsmanSha256 = "sha256-Ajqdff1SGjWrYMEwWAIrao1Al1gKEFV1A7jqPtk/2i4=";
    outputHash = "sha256-eQLiYP7pkyjX+rUvM2fBp3pgDmtgzUjEqMJZHgA72iw=";
  };
  wineWrapper =
    triple: installPath: includePaths: libPaths:
    let
      pathList =
        paths:
        builtins.concatStringsSep ";" (builtins.map (entry: "%PREFIX%/${installPath}/${entry}") paths);
    in
    (pkgs.writeShellScript "wine-wrapper.sh" ''
      set -euo pipefail
      set -x

      export INCLUDE="${pathList includePaths}"
      export LIB="${pathList libPaths}"

      # Make sure we do not spawn any window (avoid lag)
      export DISPLAY=

      # Suppress wine debug information
      export WINEDEBUG="fixme-all,warn-all"

      # Disable .Net and mshtml
      export WINEDLLOVERRIDES="mscoree,mshtml="

      # Use custom wineprefix
      export WINEPREFIX=$(TMPDIR="$PWD" mktemp -d)

      trap '${pkgs.wine}/bin/wineserver --kill || true; rm -rf "$WINEPREFIX"' EXIT

      ${pkgs.wine}/bin/wine %EXECUTABLE% "$@"

    '');
  msvcPackager =
    {
      triple,
      binPath,
      libPaths,
      includePaths,
      extract,
      useDebugInfo,
      src,
    }:
    (pkgs.stdenvNoCC.mkDerivation {
      pname = triple;
      version = "1.0";
      __structuredAttrs = true;

      # Every toolchain output embeds Microsoft VC / SDK binaries, so
      # route these to the private binary cache rather than the public
      # one.
      useCache = "private";

      # Required for ntlm_auth
      buildInputs = [ pkgs.samba ];

      src = src;

      dontUnpack = true;

      buildPhase =
        let
          installPath = "lib/vc/${triple}";
          fullBinPath = "${installPath}/${binPath}";
          currentWineWrapper = wineWrapper triple installPath includePaths libPaths;
        in
        ''
          function log() {
              echo "$1" > /dev/stderr
          }

          function quiet() {
              OUTPUT_PATH=$(mktemp)
              if ! "$@" >& "$OUTPUT_PATH"; then
                  cat "$OUTPUT_PATH"
                  rm -f "$OUTPUT_PATH"
                  exit 1
              fi
              rm -f "$OUTPUT_PATH"
          }


          mkdir -p "$out/${installPath}"
          pushd "$out/${installPath}" >& /dev/null

          log "Extracting"
          ${extract}

          log "Fixing directory permissions"
          find -type d -print0 | xargs -0 -n1000 chmod a+rx

          log "Fixing file permissions"
          find -type f -print0 | xargs -0 -n1000 chmod a+r

          mkdir "$out/bin"

          cd "${binPath}"
          shopt -s nocaseglob
          EXECUTABLES=$(echo *exe)

          MSPDBSRV=""
          if test -e "$(echo mspdbsrv.ex*e)"; then
              MSPDBSRV=$(readlink -f $(echo mspdbsrv.ex*e))
          fi

          shopt -u nocaseglob
          cd -

          for EXECUTABLE in $EXECUTABLES; do
              LOWERCASE="''${EXECUTABLE,,}"
              SUFFIX="''${LOWERCASE%.exe}"

              WRAPPER_EXECUTABLE="$out/bin/${triple}-$SUFFIX"

              log "Creating wrapper for ${triple}-$SUFFIX"

              cp -a ${currentWineWrapper} "$WRAPPER_EXECUTABLE"

              sed -e 's|%EXECUTABLE%|'"$out"'/${fullBinPath}/'"$EXECUTABLE"'|g' \
                  -e 's|%PREFIX%|'"$out"'|g' \
                  -i \
                  "$WRAPPER_EXECUTABLE"

              if test "$SUFFIX" == "cl" && test -e "$MSPDBSRV"; then
                  sed 's|%MSPDBSRV%|'"$MSPDBSRV"'|g' -i "$WRAPPER_EXECUTABLE"
              else
                  sed 's|%MSPDBSRV%||g' -i "$WRAPPER_EXECUTABLE"
              fi

              chmod +x "$WRAPPER_EXECUTABLE"

              if test "$SUFFIX" == ml64; then
                  ln -s "$WRAPPER_EXECUTABLE" "$out/bin/${triple}-ml"
              fi

          done

          popd >& /dev/null
        '';

      doCheck = true;

      checkPhase =
        let
          platform = builtins.elemAt (builtins.match "([^-]*)-.*" triple) 0;
          targetWine =
            if pkgs.hostPlatform.linuxArch == "x86_64" && platform == "i386" then
              "${pkgs.wine}/bin/wine"
            else if pkgs.hostPlatform.linuxArch == "x86_64" && platform == "x86_64" then
              "${pkgs.wine64}/bin/wine64"
            else if pkgs.hostPlatform.linuxArch == platform then
              "${pkgs.wine}/bin/wine"
            else
              "true";
        in
        ''
          log "Creating test.c"
          cat > test.c <<'EOF'
          #include <stdio.h>

          int main() {
            puts("Hello world!");
            return 0;
          }
          EOF

          log "Compiling test.c"
          export XDG_CACHE_HOME="$PWD/.cache"
          mkdir -p "$XDG_CACHE_HOME"
          $out/bin/${triple}-cl test.c -nologo ${pkgs.lib.optionalString useDebugInfo "/DEBUG:FASTLINK /Zi"}
          rm -rf "$XDG_CACHE_HOME"

          log "Inspecting test.exe file type"
          ${pkgs.file}/bin/file test.exe

          log "Running test.exe"
          export WINEPREFIX="$PWD/.wine"
          mkdir -p "$WINEPREFIX"
          ${targetWine} test.exe
          rm -rf "$WINEPREFIX"
        '';
    });
  vs17Configuration =
    targetArch:
    let
      sdkVersion = "10.0.22621.0";
      sdkComponent = "Win11SDK_10.0.22621";
      vcPath = "Contents/VC/Tools/MSVC/14.37.32822";

      targetPath =
        if targetArch == "x86_64" then
          "x64"
        else if targetArch == "i386" then
          "x86"
        else
          "arm64";
    in
    {
      triple = "${targetArch}-winsdk-vc19";

      useDebugInfo = true;

      src = null;

      extract = ''
        log "Extracting all the *.vsix files"
        mkdir extracted-vsix
        cd extracted-vsix
        find ${vs17Payloads} -name '*.vsix' | while read -r VSIX_FILE; do
            log "Extracting $VSIX_FILE"
            ${pkgs.python3}/bin/python -c 'import zipfile; zipfile.ZipFile("'"$VSIX_FILE"'").extractall(".")'
        done
        cd ..

        mv extracted-vsix/${vcPath} VC
        rm -rf extracted-vsix

        log "Extracting all the *.cab of the SDK"
        mkdir extracted-cabs
        cd extracted-cabs
        find ${vs17Payloads}/${sdkComponent}/Installers -name '*.cab' | while read -r CAB; do
            log "Extracting $CAB"
            yes S 2>/dev/null | quiet ${pkgs.p7zip}/bin/7z x "$CAB" || true
        done

        log "Extracting meta-information from MSI installers"
        find ${vs17Payloads}/${sdkComponent}/Installers -name '*.msi' | while read -r MSI; do
            log "Extracting $MSI"
            ${pkgs.msitools}/bin/msiinfo export "$MSI" Directory | sed 's/\t/,/g' > dirs.csv
            ${pkgs.msitools}/bin/msiinfo export "$MSI" File | sed 's/\t/,/g' > file.csv
            ${pkgs.msitools}/bin/msiinfo export "$MSI" Component | sed 's/\t/,/g' > component.csv

            if test -e all-dirs.csv; then
                cat dirs.csv | tail -n +2 >> all-dirs.csv
                cat file.csv | tail -n +2 >> all-file.csv
                cat component.csv | tail -n +2 >> all-component.csv
                rm dirs.csv file.csv component.csv
            else
                mv dirs.csv all-dirs.csv
                mv file.csv all-file.csv
                mv component.csv all-component.csv
            fi
        done

          log "Extracting MSI installers"
          ${pkgs.python3}/bin/python ${./vs10-create-directories.py} \
            --create-directories \
            all-dirs.csv \
            all-file.csv \
            all-component.csv | \
            while IFS=, read -r TARGET_NAME TARGET_PATH; do
                if ! test -e "$TARGET_NAME"; then
                    echo "Warning: $TARGET_NAME not found" > /dev/stderr
                    continue
                fi

                mv "$TARGET_NAME" "./$TARGET_PATH"
            done

          cd ..

          mv 'extracted-cabs/SourceDir/Windows Kits/10' windows-sdk

          rm -rf extracted-cabs/
        '';

      binPath = "VC/bin/Hostx86/${targetPath}";
      includePaths = [
        "VC/include"
        "windows-sdk/Include/${sdkVersion}/ucrt"
        "windows-sdk/Include/${sdkVersion}/um"
        "windows-sdk/Include/${sdkVersion}/shared"
      ];
      libPaths = [
        "VC/lib/${targetPath}"
        "windows-sdk/Lib/${sdkVersion}/um/${targetPath}"
        "windows-sdk/Lib/${sdkVersion}/ucrt/${targetPath}"
      ];
    };
  toolchains = [
  # cl.exe version 12.00.8168
  # link.exe version 6.00.8168
  # Visual C++ from Visual Studio 6 (1998)
  # SDK version 5.0.1636.1
  # 32-bit host, 32-bit target
  (msvcPackager {
    triple = "i386-winsdk-vc12";

    useDebugInfo = true;

    src = fetchPrivateUrl {
      url = "https://archive.org/download/vsp600enu/VSP600ENU1.iso";
      hash = "sha256-JjTKRIFjU5h4t8QFwR7zWb8N3E8F0muVdeC5+Bn2Qn8=";
    };

    extract = ''
      quiet ${pkgs.p7zip}/bin/7z \
        x \
        "$src" \
        VC98/ \
        COMMON/MSDEV98/BIN/MSPDB60.DLL

      mv COMMON/MSDEV98/BIN/MSPDB60.DLL VC98/BIN/
    '';

    binPath = "VC98/BIN";
    includePaths = [ "VC98/INCLUDE" ];
    libPaths = [ "VC98/LIB" ];
  })

  # cl.exe version 13.10.3077
  # link.exe version 7.10.3077
  # Visual C++ from Visual Studio 7.1 (.NET 2003)
  # Unknown SDK version, probably around 5.1 or 5.2
  # 32-bit host, 32-bit target
  (msvcPackager {
    triple = "i386-winsdk-vc13";

    useDebugInfo = false;

    src = fetchPrivateUrl {
      url = "https://archive.org/download/microsoft-visual-studio-.-net-2003-professional-disc-1/Microsoft%20Visual%20Studio%20.NET%202003%20Professional%20-%20Disc%201.iso";
      hash = "sha256-oCjJiZ6avBb57fcJoPrNyP02z/d1JsPtMnxGXGxIgzc=";
    };

    extract = ''
      quiet ${pkgs.p7zip}/bin/7z \
        x \
        "$src" \
        'Program Files/Microsoft Visual Studio .NET 2003/Vc7' \
        'Program Files/Microsoft Visual Studio .NET 2003/Common7/IDE/mspdb71.dll'

      cp \
        'Program Files/Microsoft Visual Studio .NET 2003/Common7/IDE/mspdb71.dll' \
        'Program Files/Microsoft Visual Studio .NET 2003/Vc7/bin/'

      mv \
        'Program Files/Microsoft Visual Studio .NET 2003/Vc7' \
        .

      rm -rf 'Program Files'
    '';

    binPath = "Vc7/bin";
    includePaths = [ "Vc7/include" ];
    libPaths = [ "Vc7/lib" ];
  })

  # cl.exe version 16.00.30319.01
  # link.exe version 10.00.30319.01
  # Visual C++ from Visual Studio 10.0 (2010)
  # SDK version 6.1.7600.16385 (7.0a)
  # 32-bit host, 32-bit target
  (msvcPackager {
    triple = "i386-winsdk-vc16";

    useDebugInfo = true;

    src = fetchPrivateUrl {
      url = "https://archive.org/download/vs2010_202102/vs2010.zip";
      hash = "sha256-Hx1zM9ZMui1H52gkW8eqUb9MCp1B/T9OQUkU/Xn5VyA=";
    };

    extract = ''
      mkdir extract
      cd extract
      ${pkgs.unzip}/bin/unzip "$src"

      quiet ${pkgs.p7zip}/bin/7z x VCExpress/Ixpvc.exe

      ${pkgs.msitools}/bin/msiinfo export vs_setup.msi Directory | sed 's/\t/,/g' > dirs.csv
      ${pkgs.msitools}/bin/msiinfo export vs_setup.msi File | sed 's/\t/,/g' > file.csv
      ${pkgs.msitools}/bin/msiinfo export vs_setup.msi Component | sed 's/\t/,/g' > component.csv

      mkdir extract-vs-setup
      cd extract-vs-setup
      quiet ${pkgs.p7zip}/bin/7z x ../vs_setup.cab
      cd ..

      rm -rf SourceDir/
      ${pkgs.python3}/bin/python3 ${./vs10-create-directories.py} --create-directories dirs.csv file.csv component.csv | while IFS=, read -r TARGET_NAME TARGET_PATH; do
          if ! test -e "extract-vs-setup/$TARGET_NAME"; then
              echo "Warning: $TARGET_NAME not found" > /dev/stderr
              continue
          fi

          mv "extract-vs-setup/$TARGET_NAME" "./$TARGET_PATH"
      done

      rm -rf extract-vs-setup

      cd 'SourceDir/Program Files/Microsoft Visual Studio 10.0/Common7/IDE/'
      find . -maxdepth 1 -type f -exec cp -a {} ../../VC/bin/ \;
      cd -

      mv \
        'SourceDir/Program Files/Microsoft Visual Studio 10.0/VC' \
        ../VC

      mv \
        'SourceDir/Program Files/Microsoft SDKs/Windows/v7.0A' \
        ../sdk-70a

      cd ..

      rm -rf extract
    '';

    binPath = "VC/bin/";
    includePaths = [
      "VC/include"
      "sdk-70a/Include"
    ];
    libPaths = [
      "VC/lib"
      "sdk-70a/Lib"
    ];
  })

  # cl.exe version 19.37.32822
  # link.exe version 14.37.32822.0
  # Visual C++ v143 from Visual Studio 2022
  # SDK version 10.0.22621 (Windows 11)
  # 32-bit host, 32-bit target
  (msvcPackager (vs17Configuration "i386"))
  # 32-bit host, 64-bit target
  (msvcPackager (vs17Configuration "x86_64"))
  # 32-bit host, 64-bit ARM target
  (msvcPackager (vs17Configuration "aarch64"))
  ];
in
{
  inherit toolchains;
  byTriple = lib.listToAttrs (map (drv: { name = drv.pname; value = drv; }) toolchains);
}
