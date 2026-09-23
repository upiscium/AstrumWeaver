{ lib, buildEnv, python312, astrumweaver, integration }:

let
  pythonEnv = python312.withPackages (_: [ astrumweaver ]);
in
buildEnv {
  name = "astrumweaver-worker-support-0.1.0-dev";
  paths = [
    pythonEnv
    integration
  ];
  meta = {
    description = "AstrumWeaver Worker runtime support closure";
    license = lib.licenses.bsd3;
    platforms = lib.platforms.linux;
  };
}
