{ lib, buildEnv, stdenvNoCC, python312, python312Packages, astrumweaver, integration }:

let
  pythonEnv = python312.withPackages (_: [
    astrumweaver
    python312Packages.psycopg
  ]);

  migrations = stdenvNoCC.mkDerivation {
    pname = "astrumweaver-control-assets";
    version = "0.1.0-dev";
    src = ../.;
    dontBuild = true;
    installPhase = ''
      mkdir -p "$out/share/astrumweaver/migrations"
      cp migrations/*.sql "$out/share/astrumweaver/migrations/"
    '';
  };
in
buildEnv {
  name = "astrumweaver-control-support-0.1.0-dev";
  paths = [
    pythonEnv
    migrations
    integration
  ];
  meta = {
    description = "AstrumWeaver Control Plane runtime support closure";
    license = lib.licenses.bsd3;
    platforms = lib.platforms.linux;
  };
}
