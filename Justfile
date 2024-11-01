set dotenv-load
set positional-arguments

EMCIE_HOME := "./cache"
SERVER_ADDRESS := env("SERVER_ADDRESS", "http://localhost:8000")

setup-cache:
  mkdir -p {{EMCIE_HOME}}

@server *args: setup-cache
  EMCIE_HOME={{EMCIE_HOME}} poetry run emcie-server run {{args}}

@client *args='':
  poetry run emcie -s {{SERVER_ADDRESS}} "$@"

@chat *args='':
  poetry run emcie -s {{SERVER_ADDRESS}} agent chat "$@"

@validate *args: setup-cache
  EMCIE_HOME={{EMCIE_HOME}} poetry run emcie-server check {{args}}

test-ci-initial-server:
  mkdir -p logs
  poetry run pytest -v --plan=initial --tap-combined --tap-outdir=logs --capture=tee-sys --junit-xml=logs/testresults.xml | tee logs/testresults.log
  EXIT1=${PIPESTATUS[0]}
  echo $EXIT1

test-ci-initial-sdk:
  cd ../sdk
  mkdir -p logs
  poetry run pytest -v --plan=initial --tap-combined --tap-outdir=logs --capture=tee-sys --junit-xml=logs/testresults.xml | tee logs/testresults.log
  EXIT1=${PIPESTATUS[0]}
  echo $EXIT1

test: test-ci-initial-server test-ci-initial-sdk
