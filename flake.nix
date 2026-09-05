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
           hypothesis
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

        # Everything the test suite actually needs, and nothing else.
        #
        # `pythonEnv` above carries torch/accelerate/transformers for the demo
        # entrypoints. accelerate's own build-time check is numerically flaky
        # (it asserts two CPU gradients differ and they agree to 3 decimals), so
        # building the default shell can fail for reasons that have nothing to
        # do with this repository -- and then there is no way to run the tests
        # at all. The test shell depends on none of that.
        testPython = python.withPackages (ps: with ps; [
          pip
          setuptools
          wheel
          pytest
          hypothesis
          numpy
        ]);

      in
      {
        # `nix develop .#test` -- the shell the test suite runs in.
        #
        # aufbau is built from the sibling checkout rather than fetched from
        # PyPI: the grammars under test are the ones in ../aufbau, and a
        # published wheel would silently test a different grammar. The build is
        # skipped when the installed version already matches ../aufbau's
        # Cargo.toml, so this is a one-time cost per version bump.
        devShells.test = pkgs.mkShell {
          buildInputs = [
            testPython
            pkgs.maturin
            pkgs.rustc
            pkgs.cargo
            pkgs.pkg-config
            pkgs.openssl
            pkgs.stdenv.cc.cc.lib
            # The graders compile what they grade. Without these the corpus
            # tests do not fail -- they raise MissingCompiler, which is easy to
            # read as an environment quirk rather than as untested coverage.
            pkgs.gcc
            pkgs.ocaml
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
            export P7_TEST_VENV="$PWD/.venv-nix-test"

            # Recreate when the Nix interpreter changes, not just when the venv
            # is missing: `--system-site-packages` binds to the base prefix at
            # creation time, so a package added to `testPython` stays invisible
            # to a venv made against the previous one.
            nix_python_prefix="$(python -c 'import sys; print(sys.prefix)')"
            venv_base_prefix="$("$P7_TEST_VENV/bin/python" -c 'import sys; print(sys.base_prefix)' 2>/dev/null || true)"
            if [ ! -x "$P7_TEST_VENV/bin/python" ] || [ "$nix_python_prefix" != "$venv_base_prefix" ]; then
                rm -rf "$P7_TEST_VENV"
                python -m venv --system-site-packages "$P7_TEST_VENV"
            fi
            export VIRTUAL_ENV="$P7_TEST_VENV"
            source "$VIRTUAL_ENV/bin/activate"
            export PIP_DISABLE_PIP_VERSION_CHECK=1

            AUFBAU_SRC="$(cd "$PWD/../aufbau" 2>/dev/null && pwd || true)"
            if [ -n "$AUFBAU_SRC" ]; then
                want="$(grep -m1 '^version' "$AUFBAU_SRC/Cargo.toml" | cut -d'"' -f2)"
                have="$(python -c 'import aufbau,sys; sys.stdout.write(getattr(aufbau,"__version__",""))' 2>/dev/null || true)"
                if [ "$want" != "$have" ]; then
                    echo "building aufbau $want from $AUFBAU_SRC (installed: ''${have:-none})"
                    ( cd "$AUFBAU_SRC" && maturin build --release --interpreter "$VIRTUAL_ENV/bin/python" ) \
                      && python -m pip install --quiet --force-reinstall \
                           "$(ls -t "$AUFBAU_SRC"/target/wheels/aufbau_rs-"$want"-*.whl | head -1)"
                fi
            fi

            export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"
            echo "p7 test shell: python=$(python --version 2>&1 | cut -d' ' -f2) aufbau=$(python -c 'import aufbau;print(getattr(aufbau,"__version__","?"))' 2>/dev/null || echo missing)"
          '';
        };

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
                python -m pip install --quiet 'aufbau-rs>=0.5,<0.6' 'outlines[llguidance]>=1.2.0'
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
