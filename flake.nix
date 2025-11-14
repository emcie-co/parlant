{
  description = "Parlant - Python-based AI agent framework for enterprise customer-facing use cases";

  inputs = {
    # Use nixpkgs from before fetchCargoTarball was removed (pre-25.05)
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # Patch poetry2nix's vendored pep599.nix to add riscv64 support
        # AND fix the parseABITag regex to be more permissive
        poetry2nixSrc = pkgs.runCommand "poetry2nix-patched" {} ''
          cp -r ${poetry2nix} $out
          chmod -R +w $out

          # Add riscv64 to manyLinuxTargetMachines in vendored pep599.nix
          sed -i 's/s390x = "s390x";/s390x = "s390x";\n    riscv64 = "riscv64";/' \
            $out/vendor/pyproject.nix/lib/pep599.nix || true

          # Fix parseABITag regex to handle GraalPy tags like "graalpy242_311_native"
          # The original regex: ([a-z]+)([0-9]*)_?([a-z0-9]*)
          # doesn't fully match tags with multiple underscores like graalpy242_311_native
          # New regex allows underscores in all parts: ([a-z]+)([0-9_]*)_?([a-z0-9_]*)
          sed -i 's/\[0-9\]\*/[0-9_]*/g' $out/vendor/pyproject.nix/lib/pypa.nix
          sed -i 's/\[a-z0-9\]\*/[a-z0-9_]*/g' $out/vendor/pyproject.nix/lib/pypa.nix
        '';

        poetry2nixPatched = import poetry2nixSrc { inherit pkgs; };

        inherit (poetry2nixPatched)
          mkPoetryApplication
          mkPoetryEnv
          defaultPoetryOverrides;

        # Comprehensive overrides for problematic packages
        poetryOverrides = defaultPoetryOverrides.extend (final: prev: {
          # fastapi-cli in nixpkgs 24.11 is broken - replace with PyPI wheel
          fastapi-cli = final.buildPythonPackage rec {
            pname = "fastapi-cli";
            version = "0.0.8";
            format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/67/c3/25df7f2234c53a0c2e8e76c4439f44f5e85dc7eb0f717fe5cf7ebdf4ed98/fastapi_cli-0.0.8-py3-none-any.whl";
              sha256 = "sha256-XLI85SumYdPMU9egjVhm2bZ6kYSz4DjBsKVi91LqQ4U=";
            };
            propagatedBuildInputs = with final; [ typer uvicorn ];
            doCheck = false;
          };

          # Skip cargo vendoring for Rust-based packages - use prebuilt wheels
          bcrypt = prev.bcrypt.overridePythonAttrs (old: {
            # Don't try to vendor cargo deps from wheel files
            cargoDeps = null;
            nativeBuildInputs = [];
            doCheck = false;
          });

          # Skip auto-patchelf for NVIDIA CUDA packages (optional dependencies)
          nvidia-cufile-cu12 = prev.nvidia-cufile-cu12.overridePythonAttrs (old: {
            autoPatchelfIgnoreMissingDeps = true;
            doCheck = false;
          });

          # Fix backports namespace package collisions
          backports-zstd = prev.backports-zstd.overridePythonAttrs (old: {
            # Remove the __init__.py and __pycache__ to prevent collision
            postInstall = (old.postInstall or "") + ''
              rm -rf $out/lib/python*/site-packages/backports/__init__.py
              rm -rf $out/lib/python*/site-packages/backports/__pycache__
            '';
          });

          backports-tarfile = prev.backports-tarfile.overridePythonAttrs (old: {
            postInstall = (old.postInstall or "") + ''
              rm -rf $out/lib/python*/site-packages/backports/__init__.py
              rm -rf $out/lib/python*/site-packages/backports/__pycache__
            '';
          });

          # Git dependencies need build systems
          parlant-client = prev.parlant-client.overridePythonAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [ final.poetry-core ];
          });

          pytest-timing = prev.pytest-timing.overridePythonAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [ final.poetry-core ];
          });

          # Packages needing setuptools
          contextvars = prev.contextvars.overridePythonAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
          });

          jsonfinder = prev.jsonfinder.overridePythonAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [ final.setuptools ];
          });

          # Tokenizers - skip Rust build when using wheels
          tokenizers = prev.tokenizers.overridePythonAttrs (old: {
            # Don't add Rust toolchain for wheel installs
            nativeBuildInputs = [];
            doCheck = false;
          });

          # Skip checks for packages with test issues
          mako = prev.mako.overridePythonAttrs (old: {
            doCheck = false;
          });

          pytest-bdd = prev.pytest-bdd.overridePythonAttrs (old: {
            doCheck = false;
          });

          # types-typed-ast was removed from nixpkgs (not needed for Python 3.10+)
          # Provide from PyPI to satisfy dependency resolution
          types-typed-ast = final.buildPythonPackage {
            pname = "types-typed-ast";
            version = "1.5.8.7";
            format = "wheel";
            src = pkgs.fetchurl {
              url = "https://files.pythonhosted.org/packages/ca/c4/48fa43ca8d98503a6e826cf02681dbdd494058f3e3bb9bae38722507ca4e/types_typed_ast-1.5.8.7-py3-none-any.whl";
              sha256 = "sha256-l73ZtCKPlsaQSnbhCgUDBd2ttSm9NeTYI0cR4JxBtUM=";
            };
            doCheck = false;
          };
        });

        # Build parlant application with all dependencies
        # Note: mkPoetryApplication includes the package itself + all dependencies
        parlantApp = mkPoetryApplication {
          projectDir = ./.;
          python = pkgs.python310;
          preferWheels = true;
          overrides = poetryOverrides;
        };

        # Use the application's Python environment for running examples/server
        # This includes parlant + all its dependencies
        pythonEnv = parlantApp.dependencyEnv;

        # Dev environment = poetry environment with dev dependencies
        # This includes parlant source + all runtime deps + all dev deps (pytest, mypy, ipython, etc.)
        devPythonEnv = mkPoetryEnv {
          projectDir = ./.;
          python = pkgs.python310;
          preferWheels = true;
          overrides = poetryOverrides;
          # editablePackageSources allows using the local source instead of the built package
          # Point to src/ since that's where the parlant package is located
          editablePackageSources = {
            parlant = ./src;
          };
        };

        # Helper for example apps
        makeExampleApp = { name, script, description }:
          pkgs.writeShellScriptBin name ''
            set -e
            echo "🤖 ${description}"
            echo ""
            echo "Environment variables:"
            echo "  OPENAI_API_KEY - Required"
            echo "  OPENAI_BASE_URL - Optional (e.g., https://openrouter.ai/api/v1)"
            echo ""
            exec ${pythonEnv}/bin/python ${./examples}/${script} "$@"
          '';

      in
      {
        packages = {
          default = parlantApp;
          python-env = pythonEnv;
          dev-python-env = devPythonEnv;

          parlant-cli = pkgs.writeShellScriptBin "parlant" ''
            exec ${pythonEnv}/bin/parlant "$@"
          '';

          parlant-server = pkgs.writeShellScriptBin "parlant-server" ''
            set -e
            echo "🚀 Starting Parlant server..."
            echo "Server: http://localhost:8800 (default)"
            echo ""
            echo "Environment variables:"
            echo "  OPENAI_API_KEY - Required"
            echo "  OPENAI_BASE_URL - Optional"
            echo ""
            exec ${pythonEnv}/bin/parlant-server "$@"
          '';

          travel-agent = makeExampleApp {
            name = "parlant-travel-agent";
            script = "travel_voice_agent.py";
            description = "Parlant Travel Agent - Flight booking example";
          };

          healthcare = makeExampleApp {
            name = "parlant-healthcare";
            script = "healthcare.py";
            description = "Parlant Healthcare Agent - Scheduling example";
          };
        };

        apps = {
          default = { type = "app"; program = "${self.packages.${system}.parlant-cli}/bin/parlant"; };
          server = { type = "app"; program = "${self.packages.${system}.parlant-server}/bin/parlant-server"; };
          cli = { type = "app"; program = "${self.packages.${system}.parlant-cli}/bin/parlant"; };
          travel-agent = { type = "app"; program = "${self.packages.${system}.travel-agent}/bin/parlant-travel-agent"; };
          healthcare = { type = "app"; program = "${self.packages.${system}.healthcare}/bin/parlant-healthcare"; };

          test = { type = "app"; program = toString (pkgs.writeShellScript "run-tests" ''
            set -e
            echo "🧪 Running tests..."
            # Don't cd to the nix store source (read-only)
            # pytest will find tests in the current directory
            exec ${devPythonEnv}/bin/pytest "$@"
          ''); };

          lint = { type = "app"; program = toString (pkgs.writeShellScript "run-lint" ''
            set -e
            echo "🔍 Running linters..."
            cd ${./.}
            exec ${devPythonEnv}/bin/python scripts/lint.py --mypy --ruff "$@"
          ''); };

          format = { type = "app"; program = toString (pkgs.writeShellScript "run-format" ''
            set -e
            echo "✨ Formatting code..."
            cd ${./.}
            exec ${devPythonEnv}/bin/ruff format "$@"
          ''); };

          repl = { type = "app"; program = toString (pkgs.writeShellScript "run-repl" ''
            set -e
            echo "🐍 Python REPL with Parlant"
            echo "Import: import parlant"
            echo ""
            exec ${devPythonEnv}/bin/ipython "$@"
          ''); };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [ devPythonEnv pkgs.poetry pkgs.git pkgs.ruff ];

          shellHook = ''
            echo "🤖 Parlant development environment"
            echo "Python: $(python --version)"
            echo ""
            echo "Commands:"
            echo "  nix run .#server        - Start server"
            echo "  nix run .#travel-agent  - Travel agent example"
            echo "  nix run .#healthcare    - Healthcare example"
            echo "  nix run .#test          - Run tests"
            echo "  nix run .#lint          - Run linters"
            echo ""
            echo "Environment:"
            echo "  export OPENAI_API_KEY=your-key-here"
            echo ""
          '';
        };

        checks = {
          build = parlantApp;
          python-env-builds = pythonEnv;
          dev-env-builds = devPythonEnv;
        };
      }
    );
}
