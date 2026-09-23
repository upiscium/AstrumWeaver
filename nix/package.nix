{ lib, python312Packages }:

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
    fastapi
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
