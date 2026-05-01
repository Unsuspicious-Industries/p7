{
  description = "Proposition 7 - Type-aware constrained decoding for LLMs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          # Explicitly disable CUDA to use CPU-only packages
          config = {
            allowUnfree = true;
            cudaSupport = false;
          };
        };

        # Python with pre-built packages (no compilation)
        python = pkgs.python312;
        
        # Python environment with all dependencies needed for local dev,
        # the Flask demo backend, and most runtime entrypoints.
        # Note: aufbau-rs is fetched from PyPI via pip in shellHook
        pythonEnv = python.withPackages (ps:
          with ps;
          [
          # Build tools
          pip
          setuptools
          wheel
          
          # Development
          pytest
          numpy
          accelerate
          ipykernel
          flask
          flask-cors
          sentencepiece

          # Transformers (CPU version)
          torch
          transformers
          tokenizers
          huggingface-hub
          safetensors

          # Other useful deps
          tqdm
          pyyaml
          regex
        ]);

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Frontend toolchain
            pkgs.nodejs_20

            # Python with all packages
            pythonEnv
            
            # Build essentials
            pkgs.pkg-config
            pkgs.openssl
            pkgs.git
            pkgs.curl
            pkgs.maturin

            # For linking
            pkgs.stdenv.cc.cc.lib
          ];

          shellHook = ''
            # Set library path for linking
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
            
            # Create venv if it doesn't exist (for writable pip installs)
            if [ ! -d "$PWD/.venv" ]; then
                python -m venv "$PWD/.venv"
            fi
            export VIRTUAL_ENV="$PWD/.venv"
            source "$VIRTUAL_ENV/bin/activate"
            
            # Upgrade pip first
            pip install --quiet --upgrade pip
            
            # Install aufbau-rs from PyPI and all p7 dependencies
            pip install --quiet 'aufbau-rs>=0.1.2'
            pip install --quiet torch transformers accelerate sentencepiece huggingface-hub safetensors tokenizers modal
              
            # Install proposition7 in editable mode
            pip install --quiet -e "$PWD[modal]"
            
            # Automatically include current directory in PYTHONPATH for local dev
            export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"
            
            echo "p7 dev shell: python=$(python --version 2>&1 | cut -d' ' -f2) cuda=off"
          '';

          # Prevent Nix from trying to build CUDA packages
          CUDA_VISIBLE_DEVICES = "";
        };

        # Package for building the wheel
        packages.default = pkgs.python312Packages.buildPythonPackage {
          pname = "proposition-7";
          version = "0.1.0";
          format = "pyproject";
          
          src = ./.;
          
          nativeBuildInputs = [
            pkgs.setuptools
            pkgs.wheel
          ];
          
          buildInputs = [
            pkgs.openssl
          ];
          
          propagatedBuildInputs = with python.pkgs; [
            numpy
          ];
          
          # Skip tests during build
          doCheck = false;
        };
      }
    );
}
