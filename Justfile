set dotenv-load
set positional-arguments

PARLANT_HOME := "./parlant-data"
LOGS_DIR := "./logs"
SERVER_ADDRESS := env("SERVER_ADDRESS", "http://localhost:8800")

@unknown:
  echo "Please specify a command"


setup-cache:
  mkdir -p {{PARLANT_HOME}}

setup-logdir:
  mkdir -p {{LOGS_DIR}}


@server *args: setup-cache
  PARLANT_HOME={{PARLANT_HOME}} poetry run parlant-server {{args}}

@client *args='':
  poetry run parlant -s {{SERVER_ADDRESS}} "$@"

@chat *args='':
  poetry run parlant -s {{SERVER_ADDRESS}} agent chat "$@"


@kill-server:
  lsof -i:8800 | grep :8800 | cut -d " " -f 3 | xargs kill && echo "KILLED" || echo "NOT KILLED"

@kill-test-server:
  lsof -i:8089 | grep :8089 | cut -d " " -f 3 | xargs kill && echo "KILLED" || echo "NOT KILLED"

@kill-test-plugin-server:
  lsof -i:8091 | grep :8091 | cut -d " " -f 3 | xargs kill && echo "KILLED" || echo "NOT KILLED"
  lsof -i:8092 | grep :8092 | cut -d " " -f 3 | xargs kill && echo "KILLED" || echo "NOT KILLED"

@kill: kill-test-plugin-server kill-test-server
  echo "killed"


@test-deterministic *specs='': setup-logdir
    mkdir -p logs/deterministic
    poetry run pytest \
      -vv {{specs}} --plan=deterministic --no-cache \
      --tap-combined --tap-outdir=logs/deterministic \
      --timing-file=logs/deterministic/test_timings.csv \
      --junit-xml=logs/deterministic/testresults.xml \
      --color=auto

@test-core-stable *specs='': setup-logdir
    mkdir -p logs/core_stable
    poetry run pytest \
      -vv {{specs}} --plan=core_stable --no-cache \
      --tap-combined --tap-outdir=logs/core_stable \
      --timing-file=logs/core_stable/test_timings.csv \
      --junit-xml=logs/core_stable/testresults.xml \
      --color=auto

@test-core-unstable *specs='': setup-logdir
    mkdir -p logs/core_unstable
    poetry run pytest \
      -vv {{specs}} --plan=core_unstable --no-cache \
      --tap-combined --tap-outdir=logs/core_unstable \
      --timing-file=logs/core_unstable/test_timings.csv \
      --junit-xml=logs/core_unstable/testresults.xml \
      --color=auto

test-complete  *specs='':
  just test-deterministic {{specs}}
  just test-core-stable {{specs}}
  just test-core-unstable {{specs}}
  

@install:
  clear
  poetry lock --no-update
  poetry install

@run: install server

@regen-sdk:
  python scripts/generate_client_sdk.py

@clean:
  find . -type d | grep __pycache__ | xargs rm -rf