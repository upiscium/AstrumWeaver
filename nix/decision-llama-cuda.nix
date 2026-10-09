# Exact experimental text-only d1-3B System-One runtime. No model bundled.
# This is deliberately distinct from the generic #worker closure.
{
  lib,
  fetchFromGitHub,
  cmake,
  ninja,
  pkg-config,
  openssl,
  cudaPackages_12_9,
  autoAddDriverRunpath,
  cudaArch,
}:
let
  cuda = cudaPackages_12_9;
  sourceRevision = "bd4eeaa047006cb1fe71999fbd11134b5836e167";
in
assert builtins.elem cudaArch [ "61" "86" ];
cuda.backendStdenv.mkDerivation {
  pname = "astrumweaver-llama-decision-cuda-sm${cudaArch}";
  version = "0.6.0-dev-${builtins.substring 0 8 sourceRevision}";

  src = fetchFromGitHub {
    owner = "ggml-org";
    repo = "llama.cpp";
    rev = sourceRevision;
    hash = "sha256-R8eTkbeh47PDk/LMZUctWnrkivWCp7J0BczN6cLCBxY=";
  };

  strictDeps = true;
  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    cuda.cuda_nvcc
    autoAddDriverRunpath
  ];
  buildInputs = [
    openssl
    cuda.cccl
    cuda.cuda_cudart
    cuda.libcublas
  ];

  cmakeFlags = [
    "-DGGML_NATIVE=OFF"
    "-DGGML_CPU_ALL_VARIANTS=OFF"
    "-DGGML_CUDA=ON"
    "-DGGML_CUDA_FA=ON"
    "-DGGML_CUDA_NCCL=OFF"
    "-DGGML_BLAS=OFF"
    "-DBUILD_SHARED_LIBS=ON"
    "-DCMAKE_CUDA_ARCHITECTURES=${cudaArch}"
    "-DLLAMA_BUILD_TESTS=OFF"
    "-DLLAMA_BUILD_EXAMPLES=OFF"
    "-DLLAMA_BUILD_SERVER=ON"
    "-DLLAMA_BUILD_IS_DEV=OFF"
  ];

  # Make the source/arch pin queryable after building, including offline.
  postInstall = ''
    test -x "$out/bin/llama-server"
    mkdir -p "$out/share/astrumweaver"
    printf '%s\n' '${sourceRevision}' > "$out/share/astrumweaver/llama-cpp-revision"
    printf '%s\n' '${cudaArch}' > "$out/share/astrumweaver/cuda-sm"
    printf '%s\n' '12.9' > "$out/share/astrumweaver/cuda-toolkit"
  '';

  meta = {
    description = "Pinned llama.cpp CUDA 12.9 System-One test-only runtime, SM ${cudaArch}";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "llama-server";
  };
}
