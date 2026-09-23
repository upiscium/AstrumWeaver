{ lib, python312Packages }:

let
  # The pinned nixpkgs FastAPI package currently pulls its own upstream
  # inline-snapshot test dependency, whose test suite is broken independently
  # of AstrumWeaver. AstrumWeaver consumes FastAPI as a runtime dependency only,
  # so strip FastAPI's check-only inputs rather than overriding inline-snapshot
  # globally.
  fastapiRuntime = python312Packages.fastapi.overridePythonAttrs (_: {
    doCheck = false;
    nativeCheckInputs = [ ];
  });
in
python312Packages.buildPythonPackage {
  pname = "astrumweaver";
  version = "0.1.0-dev";
  pyproject = true;

  src = lib.cleanSourceWith {
    src = ../.;
    filter = path: type:
      let
        base = baseNameOf path;
      in
      !(base == ".git" || base == ".worktrees" || base == "__pycache__");
  };

  build-system = [ python312Packages.hatchling ];

  dependencies = with python312Packages; [
    fastapiRuntime
    httpx
    pydantic
    uvicorn
  ];

  pythonImportsCheck = [
    "astrumweaver"
    "astrumweaver.control"
  ];

  doCheck = false;

  meta = {
    description = "Deployment-agnostic compute fabric for heterogeneous compute nodes";
    homepage = "https://github.com/upiscium/AstrumWeaver";
    license = lib.licenses.bsd3;
    platforms = lib.platforms.linux;
  };
}
