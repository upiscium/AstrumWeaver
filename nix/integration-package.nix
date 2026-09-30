{ stdenvNoCC, lib, bash, python312 }:

stdenvNoCC.mkDerivation {
  pname = "astrumweaver-integration";
  version = "0.1.0-dev";

  src = ../.;
  dontBuild = true;
  dontPatchShebangs = true;

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
    cp libexec/gpu-device-map "$assetRoot/libexec/"
    cp libexec/gpu-isolation-probe "$assetRoot/libexec/"
    cp src/astrumweaver/gpu_mapping.py "$assetRoot/libexec/"
    cp libexec/worker-gpu-uuids "$assetRoot/libexec/"

    substituteInPlace "$assetRoot/libexec/worker-gpu-uuids" \
      --replace-fail "#!/usr/bin/env python3" "#!${python312}/bin/python3"

    chmod +x \
      "$assetRoot/setup/setup-control-plane.sh" \
      "$assetRoot/setup/setup-gpu-worker.sh" \
      "$assetRoot/libexec/gpu-preflight" \
      "$assetRoot/libexec/gpu-device-map" \
      "$assetRoot/libexec/gpu-isolation-probe" \
      "$assetRoot/libexec/worker-gpu-uuids"

    cat >"$out/bin/astrumweaver-setup-control-plane" <<EOF
    #!${bash}/bin/bash
    exec ${bash}/bin/bash "$assetRoot/setup/setup-control-plane.sh" "\$@"
    EOF

    cat >"$out/bin/astrumweaver-setup-gpu-worker" <<EOF
    #!${bash}/bin/bash
    exec ${bash}/bin/bash "$assetRoot/setup/setup-gpu-worker.sh" "\$@"
    EOF

    cat >"$out/bin/astrumweaver-gpu-device-map" <<EOF
    #!${bash}/bin/bash
    export ASTRUMWEAVER_GPU_MAPPING_PYTHON="$assetRoot/libexec/gpu_mapping.py"
    export ASTRUMWEAVER_GPU_MAPPING_PYTHON_BIN="${python312}/bin/python3"
    exec ${bash}/bin/bash "$assetRoot/libexec/gpu-device-map" "\$@"
    EOF

    cat >"$out/bin/astrumweaver-gpu-preflight" <<EOF
    #!${bash}/bin/bash
    exec ${bash}/bin/bash "$assetRoot/libexec/gpu-preflight" "\$@"
    EOF

    cat >"$out/bin/astrumweaver-gpu-isolation-probe" <<EOF
    #!${bash}/bin/bash
    export ASTRUMWEAVER_GPU_DEVICE_MAP_COMMAND="$out/bin/astrumweaver-gpu-device-map"
    exec ${bash}/bin/bash "$assetRoot/libexec/gpu-isolation-probe" "\$@"
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
