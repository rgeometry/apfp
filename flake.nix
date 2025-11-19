{
  description = "apfp library";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crane = {
      url = "github:ipetkov/crane";
    };
    rust-overlay.url = "github:oxalica/rust-overlay";
    advisory-db = {
      url = "github:RustSec/advisory-db/ce9208c0021cd8a6b66ff4b345171e8eedd0441c";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    crane,
    rust-overlay,
    advisory-db,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [rust-overlay.overlays.default crane.overlays.default];
      };

      craneLib = crane.mkLib pkgs;
      src = craneLib.cleanCargoSource (craneLib.path ./.);
      crateInfo = craneLib.crateNameFromCargoToml {cargoToml = ./Cargo.toml;};
      pname = crateInfo.pname;
      version = crateInfo.version;

      toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

      cargoArtifacts = craneLib.buildDepsOnly {
        inherit src pname version;
        nativeBuildInputs = [toolchain];
      };

      package = craneLib.buildPackage {
        inherit src pname version cargoArtifacts;
        nativeBuildInputs = [toolchain];
      };

      commonArgs = {
        inherit src pname version cargoArtifacts;
        nativeBuildInputs = [toolchain];
      };

      cargoDocCheck = craneLib.cargoDoc commonArgs;

      cargoClippyCheck =
        craneLib.cargoClippy (commonArgs // {cargoClippyExtraArgs = "--all-targets -- --deny warnings";});

      cargoFmtCheck = craneLib.cargoFmt {inherit src;};

      cargoTaploCheck = craneLib.taploFmt {
        src = pkgs.lib.sources.sourceFilesBySuffices src [".toml"];
      };

      cargoNextestCheck = craneLib.cargoNextest commonArgs;

      formatApp = pkgs.writeShellApplication {
        name = "apfp-format";
        runtimeInputs = [
          toolchain
          pkgs.taplo
          pkgs.alejandra
        ];
        text = ''
          set -euo pipefail
          cargo fmt
          taplo fmt
          alejandra .
        '';
      };

      fmtCheck =
        pkgs.runCommand "alejandra-check" {
          nativeBuildInputs = [pkgs.alejandra];
        } ''
          alejandra --check ${./flake.nix}
          touch $out
        '';

      cargoAuditCheck = craneLib.cargoAudit {
        inherit src pname version;

        "advisory-db" = advisory-db;
        nativeBuildInputs = [toolchain];
      };

      # Pinned target for consistent assembly output across platforms
      asmTarget = "aarch64-unknown-linux-gnu";

      # Functions to check for assembly output
      asmFunctions = [
        "apfp::analysis::ast_static::orient2d_fast"
        "apfp::analysis::ast_static::cmp_dist_fast"
      ];

      cargoAsmCheck =
        pkgs.runCommand "cargo-asm-check" {
          nativeBuildInputs = [
            toolchain
          ];
        } ''
          set -euo pipefail

          # Copy source
          cp -r ${src}/* .
          chmod -R +w .

          # Copy built artifacts if available (for faster rebuilds)
          if [ -d "${package}/lib" ] || [ -d "${package}/target" ]; then
            mkdir -p target
            # Try to reuse artifacts if they exist
          fi

          export CARGO_TARGET_DIR=$PWD/target
          export CARGO_HOME=$PWD/.cargo

          # Install cargo-asm
          echo "Installing cargo-asm..."
          cargo install --locked --root "$CARGO_HOME" cargo-asm 2>&1 || {
            echo "Trying git fallback..."
            cargo install --locked --root "$CARGO_HOME" --git https://github.com/pacak/cargo-asm cargo-asm 2>&1
          }

          # Add cargo-asm to PATH
          export PATH="$CARGO_HOME/bin:$PATH"

          # Build the library in release mode for the pinned target
          echo "Building for target ${asmTarget}..."
          cargo build --release --target ${asmTarget} --lib

          # Patterns that indicate assertions (panic, assert, etc.)
          ASSERT_PATTERNS="panic|assert|__rust_start_panic|rust_begin_unwind"

          # Patterns that indicate memory allocations
          ALLOC_PATTERNS="__rust_alloc|__rust_realloc|__rust_dealloc|malloc|calloc|realloc|alloc::alloc"

          for func in ${pkgs.lib.concatStringsSep " " (map pkgs.lib.escapeShellArg asmFunctions)}; do
            echo "Checking assembly for function: $func"

            # Generate assembly output
            asm_output=$(cargo asm --release --target ${asmTarget} --lib "$func" 2>&1 || true)

            if [ -z "$asm_output" ]; then
              echo "ERROR: Failed to generate assembly for $func"
              exit 1
            fi

            # Check for assertions
            if echo "$asm_output" | grep -qiE "$ASSERT_PATTERNS"; then
              echo "ERROR: Function $func contains assertions in assembly:"
              echo "$asm_output" | grep -iE "$ASSERT_PATTERNS"
              exit 1
            fi

            # Check for memory allocations
            if echo "$asm_output" | grep -qiE "$ALLOC_PATTERNS"; then
              echo "ERROR: Function $func contains memory allocations in assembly:"
              echo "$asm_output" | grep -iE "$ALLOC_PATTERNS"
              exit 1
            fi

            echo "✓ Function $func passed checks (no assertions, no allocations)"
          done

          touch $out
        '';
    in {
      packages.default = package;

      checks = {
        fmt = fmtCheck;
        "cargo-fmt" = cargoFmtCheck;
        "cargo-taplo" = cargoTaploCheck;
        "cargo-doc" = cargoDocCheck;
        "cargo-clippy" = cargoClippyCheck;
        "cargo-nextest" = cargoNextestCheck;
        cargo-audit = cargoAuditCheck;
        "cargo-asm" = cargoAsmCheck;
        default = package;
      };

      devShells.default = pkgs.mkShell {
        inputsFrom = [package];
        buildInputs = [
          toolchain
          pkgs.cargo-audit
          pkgs.cargo-nextest
          pkgs.alejandra
          pkgs.taplo
          pkgs.rust-analyzer
        ];
      };

      formatter = pkgs.alejandra;

      apps.format =
        flake-utils.lib.mkApp {
          drv = formatApp;
          exePath = "/bin/apfp-format";
        }
        // {
          meta.description = "Format Rust sources, TOML files, and Nix expressions";
        };
    });
}
