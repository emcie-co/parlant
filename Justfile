set dotenv-load
set positional-arguments

PARLANT_HOME := "./cache"
LOGS_DIR := "./logs"
SERVER_ADDRESS := env("SERVER_ADDRESS", "http://localhost:8800")

what:
  echo "just... what?"

setup-cache:
  mkdir -p {{PARLANT_HOME}}

@server *args: setup-cache
  PARLANT_HOME={{PARLANT_HOME}} poetry run parlant-server {{args}}

@client *args='':
  poetry run parlant -s {{SERVER_ADDRESS}} "$@"

@chat *args='':
  poetry run parlant -s {{SERVER_ADDRESS}} agent chat "$@"

@kill-server:
  lsof -i:8800 | grep :8800 | cut -d " " -f 3 | xargs kill && echo "KILLED" || echo "NOT KILLED"

@kill-test-server:
  lsof -i:8091 | grep :8091 | cut -d " " -f 3 | xargs kill && echo "KILLED" || echo "NOT KILLED"

@kill-test-plugin-server:
  lsof -i:8089 | grep :8089 | cut -d " " -f 3 | xargs kill && echo "KILLED" || echo "NOT KILLED"

@kill: kill-test-plugin-server kill-test-server
  echo "killed"

@mklogdir:
  mkdir -p {{LOGS_DIR}}

@test *tests='': mklogdir
  poetry run pytest \
    -vv {{tests}} --plan=complete \
    --tap-combined --tap-outdir=logs \
    --timing-file=logs/test_timings.csv \
    --junit-xml=logs/testresults.xml \
    --color=auto

@test-ns *tests='': mklogdir
  poetry run pytest \
    -vv {{tests}} \
    --tap-combined --tap-outdir=logs \
    --timing-file=logs/test_timings.csv \
    --junit-xml=logs/testresults.xml \
    --color=auto

@test-deterministic *specs='':
    poetry run pytest \
      -vv {{specs}} --plan=deterministic --use-cache \
      --tap-combined --tap-outdir=logs/deterministic \
      --timing-file=logs/deterministic/test_timings.csv \
      --junit-xml=logs/deterministic/testresults.xml \
      --color=auto

@test-stochastic *specs='':
    poetry run pytest \
      -vv {{specs}} --plan=stochastic \
      --tap-combined --tap-outdir=logs/stochastic \
      --timing-file=logs/stochastic/test_timings.csv \
      --junit-xml=logs/stochastic/testresults.xml \
      --color=auto

@test-core-stable *specs='':
    poetry run pytest \
      -vv {{specs}} --plan=core_stable \
      --tap-combined --tap-outdir=logs/core_stable \
      --timing-file=logs/core_stable/test_timings.csv \
      --junit-xml=logs/core_stable/testresults.xml \
      --color=auto

@test-core-unstable *specs='':
    poetry run pytest \
      -vv {{specs}} --plan=core_unstable \
      --tap-combined --tap-outdir=logs/core_unstable \
      --timing-file=logs/core_unstable/test_timings.csv \
      --junit-xml=logs/core_unstable/testresults.xml \
      --color=auto

@test-core-experimental *specs='':
    poetry run pytest \
      -vv {{specs}} --plan=core_experimental \
      --tap-combined --tap-outdir=logs/core_experimental \
      --timing-file=logs/core_experimental/test_timings.csv \
      --junit-xml=logs/core_experimental/testresults.xml \
      --color=auto

@test-complete *specs='':
  poetry run pytest \
      -vv {{specs}} --plan=complete \
      --tap-combined --tap-outdir=logs/complete \
      --timing-file=logs/complete/test_timings.csv \
      --junit-xml=logs/complete/testresults.xml \
      --color=auto
  

@test-list:
  echo "just test-deterministic #(uses cache)"
  echo "# just test-stochastic #(no cache)"
  echo "# just test-core-stable #(no cache)"
  echo "just test-core-unstable #(no cache)"
  echo "# just test-core-experimental #(no cache)"
  
@install:
  clear
  poetry lock --no-update
  poetry install

@run: install server

@regen-sdk:
  python scripts/generate_client_sdk.py

@decache:
  find . -type d | grep __pycache__ | xargs rm -rf