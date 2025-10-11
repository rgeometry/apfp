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
