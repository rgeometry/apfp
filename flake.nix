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

      # Toolchain with target support for cross-compilation
      toolchainWithTarget = toolchain.override {
        targets = [asmTarget];
      };

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

      # Use cargo-show-asm from nixpkgs, which provides the cargo-asm program
      cargoAsmTool = pkgs.cargo-show-asm;

      # Package that generates assembly output for human inspection
      # Build using craneLib which handles vendoring, then extract assembly
      # Always targets aarch64-unknown-linux-gnu for consistency
      cargoAsmOutput = let
        # Build the package first
        builtPackage = craneLib.buildPackage {
          inherit src pname version cargoArtifacts;
          nativeBuildInputs = [toolchain];
        };
      in
        pkgs.runCommand "cargo-asm-output" {
          nativeBuildInputs = [
            toolchainWithTarget
            cargoAsmTool
          ];
          builtPackage = builtPackage;
          meta.description = "Assembly output for critical functions (for inspection)";
        } ''
          set -euo pipefail

          # Copy source
          cp -r ${src}/* .
          chmod -R +w .

          # Extract the built target directory from the package
          mkdir -p target
          if [ -f "${builtPackage}/target.tar.zst" ]; then
            echo "Extracting target directory from built package..."
            tar -xf ${builtPackage}/target.tar.zst -C target --strip-components=1 || {
              zstd -dc ${builtPackage}/target.tar.zst | tar -x -C target --strip-components=1 || true
            }
          fi

          # Set up vendored dependencies
          vendorDir=${craneLib.vendorCargoDeps {inherit src cargoArtifacts;}}

          # Copy the config.toml which contains the vendor directory path
          mkdir -p .cargo
          cp $vendorDir/config.toml .cargo/config.toml

          # The config.toml points to a symlink in vendorDir, we need to follow it
          # and copy the actual vendor registry
          vendorRegPath=$(readlink -f $vendorDir/* 2>/dev/null | grep vendor-registry | head -1)
          if [ -n "$vendorRegPath" ] && [ -d "$vendorRegPath" ]; then
            # Copy the vendor registry directory
            mkdir -p vendor
            cp -r $vendorRegPath/* vendor/ 2>/dev/null || true
          fi

          export CARGO_TARGET_DIR=$PWD/target

          # Build the library in release mode for the pinned target
          echo "Building for target ${asmTarget}..."
          cargo build --release --target ${asmTarget} --lib --offline

          # Create output directory
          mkdir -p $out

          # Generate assembly output for each function and save to files
          for func in ${pkgs.lib.concatStringsSep " " (map pkgs.lib.escapeShellArg asmFunctions)}; do
            echo "Generating assembly for function: $func"

            # Convert function name to filename (replace :: with _)
            filename=$(echo "$func" | sed 's/::/_/g')

            # Generate assembly output (using pinned target, offline mode)
            # Build messages go to stderr, assembly goes to stdout - keep them separate
            cargo asm --release --target ${asmTarget} --lib "$func" --offline > "$out/$filename.s" 2>/dev/null || {
              echo "Warning: Failed to generate assembly for $func"
              # Capture error output for debugging
              cargo asm --release --target ${asmTarget} --lib "$func" --offline > "$out/$filename.s.error" 2>&1 || true
            }

            echo "Saved assembly to $out/$filename.s"
          done

          # Create a README with information about the assembly files
          cat > $out/README.md <<EOF
          # Assembly Output

          This directory contains the generated assembly code for critical functions.

          Target: ${asmTarget}
          Build mode: release

          Functions:
          ${pkgs.lib.concatMapStringsSep "\n" (f: "- \`${f}\`") asmFunctions}

          ## Files

          ${pkgs.lib.concatMapStringsSep "\n" (f: "- \`$(echo ${f} | sed 's/::/_/g').s\` - Assembly for \`${f}\`") asmFunctions}

          ## Viewing the Assembly

          You can view these files with any text editor or use:
          \`\`\`bash
          cat \$out/*.s
          \`\`\`

          Or navigate to the package output:
          \`\`\`bash
          nix build .#cargoAsmOutput
          cat result/*.s
          \`\`\`
          EOF

          echo "Assembly output generated in $out"
        '';

      # Check scripts are in the nix/ folder
      checkNoAssertionsScript = ./nix/check-no-assertions.sh;
      checkNoAllocationsScript = ./nix/check-no-allocations.sh;

      cargoAsmCheck =
        pkgs.runCommand "cargo-asm-check" {
          nativeBuildInputs = [pkgs.bash];
          # Use the output from cargoAsmOutput
          asmOutput = cargoAsmOutput;
          checkNoAssertions = checkNoAssertionsScript;
          checkNoAllocations = checkNoAllocationsScript;
        } ''
          set -euo pipefail

          echo "Checking assembly files from cargoAsmOutput..."

          for asm_file in ${cargoAsmOutput}/*.s; do
            if [ ! -f "$asm_file" ]; then
              echo "ERROR: No assembly files found in ${cargoAsmOutput}"
              exit 1
            fi

            filename=$(basename "$asm_file")
            echo "Checking $filename..."

            # Run assertion check
            "$checkNoAssertions" "$asm_file"

            # Run allocation check
            "$checkNoAllocations" "$asm_file"

            echo "✓ $filename passed all checks"
          done

          touch $out
        '';

      # App to display assembly check results as a table
      asmCheckTableApp = pkgs.writeShellApplication {
        name = "asm-check-table";
        runtimeInputs = [pkgs.bash];
        text = ''
          set -euo pipefail

          # Get the assembly output directory
          asm_output=$(nix build --no-link --print-out-paths '.#cargoAsmOutput' 2>/dev/null || echo "")

          if [ -z "$asm_output" ] || [ ! -d "$asm_output" ]; then
            echo "Building cargoAsmOutput..."
            asm_output=$(nix build --print-out-paths '.#cargoAsmOutput' 2>/dev/null)
          fi

          # Pass function names to the script for proper filename matching
          ${./nix/asm-check-table.sh} "$asm_output" "${./nix}" ${pkgs.lib.concatStringsSep " " (map pkgs.lib.escapeShellArg asmFunctions)}
        '';
      };
    in {
      packages.default = package;
      packages.cargoAsmOutput = cargoAsmOutput;
      packages.cargoAsmTool = cargoAsmTool;

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

      apps.asmCheckTable =
        flake-utils.lib.mkApp {
          drv = asmCheckTableApp;
          exePath = "/bin/asm-check-table";
        }
        // {
          meta.description = "Display assembly check results as a table";
        };
    });
}
