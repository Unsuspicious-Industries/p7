#!/usr/bin/env bash
# The runtime environment, on NixOS. Source or use as a prefix:
#   ./env.sh python -m pytest tests/ -q
#
# This is the congen-wide convention: every repo carries an `env.sh` with this
# same contract, so the same command runs the suite anywhere in the tree. Only
# the PYTHONPATH line below differs between repos.
#
# What needs help here and nowhere else:
#   libstdc++/libz  numpy's manylinux wheel links them by soname; NixOS has no
#                   /usr/lib for the loader to find them in
#
# Note it cds to its own repo first, so `../p7/env.sh` from wirt runs *p7's*
# suite, not wirt's. That has silently reported the wrong repo's test count.
set -eu
cd "$(dirname "$0")"
exec nix shell nixpkgs#zlib nixpkgs#stdenv.cc.cc.lib --command bash -c '
  export LD_LIBRARY_PATH="$(nix eval --raw nixpkgs#zlib.out)/lib:$(nix eval --raw nixpkgs#stdenv.cc.cc.lib)/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PYTHONPATH=".:src${PYTHONPATH:+:$PYTHONPATH}"
  export PATH="$PWD/.venv/bin:$PATH"
  exec "$@"
' bash "$@"
