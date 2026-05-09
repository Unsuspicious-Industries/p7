{
  description = "Proposition 7 - Type-aware constrained decoding for LLMs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    vast-cli.url = "github:dialohq/vast-cli.nix";
    vast-cli.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, vast-cli }:
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

            # Vast.ai CLI
            vast-cli.packages.${pkgs.system}.default

            # Python with all packages
            pythonEnv
            
            # Build essentials
            pkgs.pkg-config
            pkgs.openssl
            pkgs.git
            pkgs.curl
            pkgs.jq
            pkgs.maturin

            # For linking
            pkgs.stdenv.cc.cc.lib
          ];

          shellHook = ''
            # Set library path for linking
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
            
            export PROPOSITION7_NIX_VENV="$PWD/.venv-nix"

            recreate_nix_venv() {
                rm -rf "$PROPOSITION7_NIX_VENV"
                python -m venv --system-site-packages "$PROPOSITION7_NIX_VENV"
            }

            # Keep the Nix shell isolated from any uv-managed .venv.
            if [ ! -x "$PROPOSITION7_NIX_VENV/bin/python" ] || ! "$PROPOSITION7_NIX_VENV/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1; then
                recreate_nix_venv
            fi

            export VIRTUAL_ENV="$PROPOSITION7_NIX_VENV"
            source "$VIRTUAL_ENV/bin/activate"
            export PIP_DISABLE_PIP_VERSION_CHECK=1

            ensure_nix_shell_packages() {
                if ! python -m pip --version >/dev/null 2>&1; then
                    python -m ensurepip --upgrade >/dev/null 2>&1 || true
                fi
                python -m pip install --quiet --upgrade pip 'setuptools<82' wheel build
                python -m pip install --quiet 'aufbau-rs>=0.1.2' modal 'outlines[llguidance]>=1.2.0'
                python -m pip install --quiet --no-deps -e "$PWD"
            }

            SETUP_STAMP="$VIRTUAL_ENV/.proposition7-nix-shell-v3"
            if [ ! -f "$SETUP_STAMP" ] || [ "$PWD/pyproject.toml" -nt "$SETUP_STAMP" ] || [ "$PWD/flake.nix" -nt "$SETUP_STAMP" ]; then
                if ! ensure_nix_shell_packages; then
                    recreate_nix_venv
                    export VIRTUAL_ENV="$PROPOSITION7_NIX_VENV"
                    source "$VIRTUAL_ENV/bin/activate"
                    ensure_nix_shell_packages
                fi
                touch "$SETUP_STAMP"
            fi
            
            # Automatically include current directory in PYTHONPATH for local dev
            export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"
            
            echo "proposition7 nix shell: python=$(python --version 2>&1 | cut -d' ' -f2) venv=$(basename "$VIRTUAL_ENV") cuda=off"
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
          
          nativeBuildInputs = with pkgs.python312Packages; [
            setuptools
            wheel
          ];
          
          buildInputs = [
            pkgs.openssl
          ];
          
          propagatedBuildInputs = with pkgs.python312Packages; [
            grpcio
            numpy
          ];

          # `aufbau-rs` is installed from PyPI in the dev shell rather than from nixpkgs.
          dontCheckRuntimeDeps = true;

          # Skip tests during build
          doCheck = false;
        };
      }
    );
}
