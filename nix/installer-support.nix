{ lib, buildEnv, stdenvNoCC, python312, python312Packages, astrumweaver, integration }:

let
  pythonEnv = python312.withPackages (_: [
    astrumweaver
    python312Packages.psycopg
  ]);

  migrations = stdenvNoCC.mkDerivation {
    pname = "astrumweaver-installer-assets";
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
  name = "astrumweaver-installer-support-0.1.0-dev";
  paths = [
    pythonEnv
    migrations
    integration
  ];
  meta = {
    description = "AstrumWeaver first-run installer and Control/Worker bootstrap closure";
    license = lib.licenses.bsd3;
    platforms = lib.platforms.linux;
  };
}
