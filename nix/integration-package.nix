{ stdenvNoCC, lib, bash }:

stdenvNoCC.mkDerivation {
  pname = "astrumweaver-integration";
  version = "0.1.0-dev";

  src = ../.;
  dontBuild = true;

  installPhase = ''
    runHook preInstall

    assetRoot="$out/share/astrumweaver"
    mkdir -p "$assetRoot/setup/lib" "$assetRoot/systemd" "$assetRoot/libexec" "$out/bin"

    cp setup/setup-control-plane.sh "$assetRoot/setup/"
    cp setup/setup-gpu-worker.sh "$assetRoot/setup/"
    cp setup/lib/common.sh "$assetRoot/setup/lib/"
    cp systemd/astrumweaver-control.service.in "$assetRoot/systemd/"
    cp systemd/astrumweaver-worker.service.in "$assetRoot/systemd/"
    cp libexec/gpu-preflight "$assetRoot/libexec/"

    chmod +x \
      "$assetRoot/setup/setup-control-plane.sh" \
      "$assetRoot/setup/setup-gpu-worker.sh" \
      "$assetRoot/libexec/gpu-preflight"

    cat >"$out/bin/astrumweaver-setup-control-plane" <<EOF
    #!${bash}/bin/bash
    exec ${bash}/bin/bash "$assetRoot/setup/setup-control-plane.sh" "\$@"
    EOF

    cat >"$out/bin/astrumweaver-setup-gpu-worker" <<EOF
    #!${bash}/bin/bash
    exec ${bash}/bin/bash "$assetRoot/setup/setup-gpu-worker.sh" "\$@"
    EOF

    cat >"$out/bin/astrumweaver-gpu-preflight" <<EOF
    #!${bash}/bin/bash
    exec ${bash}/bin/bash "$assetRoot/libexec/gpu-preflight" "\$@"
    EOF

    chmod +x "$out/bin/"*

    runHook postInstall
  '';

  meta = {
    description = "AstrumWeaver existing-node setup and systemd integration assets";
    license = lib.licenses.bsd3;
    platforms = lib.platforms.linux;
  };
}
