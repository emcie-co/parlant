# The Parlant CLI

The `parlant` command does four jobs: it creates projects, answers questions
about Parlant from current documentation, copies the chat widget into your
app, and drives the Parlant dashboard.

It ships with the Parlant Python package. There is nothing else to install.

```sh
pip install parlant
parlant --version
```

Every command explains itself, and every command shows worked examples at the
end of its help:

```sh
parlant --help
parlant dashboard --help
parlant dashboard session view --help
```

This guide covers what the commands are for and how they fit together. Reach
for `--help` for the exact flags.


## Starting a project

`parlant init` writes a working project you can run straight away.

```sh
parlant init my-agent
```

It asks which package manager you use (uv, pip, poetry or conda) and writes
the project to match: dependencies, a `run.sh` and `run.cmd`, and a pytest
setup. Skip the question with `--package-manager`.

Three scaffolds are available. The default, `basic`, gives you the structure
and no example code. The other two are worked agents, with tools, policies,
canned responses and tests:

- `retail-store`: a support agent with two domains, orders and disputes.
- `lead-gen`: a website agent that answers visitors' questions, qualifies
  them, scores them in code, and books meetings with a rep.

```sh
parlant init my-agent --scaffold retail-store --package-manager uv
parlant init my-agent --scaffold lead-gen --package-manager uv
```

Then:

```sh
cd my-agent
cp .env.example .env     # put your model API key in it
./run.sh
```

`DESTINATION` defaults to the current directory. Running `parlant init` with no
destination asks for confirmation first, because that is the one case you
cannot undo by deleting a folder. An existing folder is fine either way: the
command refuses only on the individual files it would overwrite, so
scaffolding next to a README or into a fresh git repository works. `--force`
overwrites them.

The project also gets an `AGENTS.md`. It tells coding agents what Parlant is,
how this project is laid out, and to consult `parlant docs query` instead of
guessing at the framework's idioms. If you already have an `AGENTS.md`, `init`
refuses rather than overwriting it.


## Asking about Parlant

`parlant docs query` answers questions from Parlant's current documentation and
cites its sources. It does not guess: when the documentation does not cover
something, it says so rather than inventing an answer.

```sh
parlant docs query "How do I add a tool to an agent?"
```

Pass `--context` with what you are building, what you have tried, and the code
it concerns. The answer is only as specific as the context you give it.

```sh
parlant docs query "Why is my policy not matching?" --context "$(cat domains/orders/domain.py)"
```

This is the command `AGENTS.md` points coding agents at, and it is worth using
yourself before a design decision rather than only after a bug.

```sh
parlant docs query "Should this be a policy or an observation?" --context "The rule is: never quote a price to an unpaid account."
```

`--json` returns the whole response — answer, citations, the documentation
revision it was drawn from, and token usage — for programs rather than people.

```sh
parlant docs query "How do I test agent behavior?" --json
```

Questions and context are sent to Parlant's documentation service and its
model provider, which keeps secret-redacted records of requests and answers
for staff review. Do not include credentials, customer data or anything from
your `.env`.


## The chat widget

`parlant widget init` copies the chat widget's source into your app. It is a
file copy, so it needs no node toolchain, and the copy is yours to change:
restyle it, swap the transport, add your own tool widgets.

```sh
parlant widget init ./web/chat
```

`DESTINATION` defaults to `parlant-chat`. The copy records the Parlant version
it came from, because its transport speaks that version's HTTP API. Re-run with
`--force` after upgrading Parlant.

To see the widget working before you commit to it, the playground runs it
against a sample agent:

```sh
parlant widget playground
parlant widget playground --port 8000
```

This one does need npm, and it needs a model API key.


## The dashboard

Dashboard commands live under `parlant dashboard`. They are implemented in
this repository, ship with the Python package, and need no separate installer.
The retired Go CLI is gone; this is the supported one.

### Signing in

```sh
parlant dashboard auth login
```

You approve this machine in your browser, for one or more workspaces. Each
workspace gets its own sign-in, and the first one you choose becomes active.
Signing in again adds workspaces, or replaces the sign-ins of the ones you
approve again.

```sh
parlant dashboard auth status
parlant dashboard workspace list
parlant dashboard workspace use WORKSPACE_ID
```

`workspace use` sets the workspace that later commands act in, so you do not
pass `--workspace` every time. Any command accepts `--workspace` to override
it, and `--project` to pick the project it acts on.

On a machine with no browser, `--no-browser` prints the URL to open elsewhere.
When a command needs a permission your sign-in does not have, `--reauthorize`
requests it without losing the sign-in you already have.

```sh
parlant dashboard auth logout
parlant dashboard auth logout --all
```

### Connecting a project you run yourself

A project you run on your own machine reaches the dashboard with a project
token. `project connect` puts that token in the folder's `.env`:

```sh
parlant dashboard project connect ./my-agent
```

`parlant init --dashboard` does both at once: it creates the project on the
dashboard first, so nothing is written locally unless the dashboard accepts
it, then connects it.

```sh
parlant init my-agent --dashboard --yes
```

Once your server is running, the dashboard has to see it before setup is
complete:

```sh
parlant dashboard project check
parlant dashboard project confirm
```

### Creating a project

```sh
parlant dashboard project create
```

Run on its own, this walks you through setup the way the dashboard does. For
managed hosting it connects or selects GitHub, then takes a repository and a
branch — the project is created from those two alone. A setup check then starts
by itself; the command follows it and shows what it found: the run and build
commands, how it classified the repository, and warnings such as environment
variables or secrets it expects. It offers all of that for review, defaulting
to what the check found, and saves your changes.

For automation, give the settings and confirm with `--yes`:

```sh
parlant dashboard project create \
  --kind managed --name MyAgent \
  --repo example/agent --branch main \
  --run-command 'python agent.py' \
  --build-command 'uv sync --frozen' \
  --build-env BUILD_MODE=production \
  --run-env APP_ENV=production \
  --secret API_KEY=MY_API_KEY \
  --yes
```

`--secret` takes the name of a local environment variable to read the value
from, not the value itself. With `--yes`, the options you gave are saved after
the check and everything else keeps what the check found.

Creating a project needs the `projects.create` permission, and `projects.update`
as well for a managed project, whose setup the command saves.

A failed or still-running check does not undo the creation. The command says so
and points at the project's setup page, and you can deploy once the settings
are right. If a repository appears under more than one GitHub installation,
`--installation` picks one. Billing and workspace readiness problems have to be
resolved in the dashboard first. Creating a project does not start a build.

The command prints a creation request ID before it submits anything. If it is
interrupted, reuse that ID:

```sh
parlant dashboard project create --request-id PRINTED_REQUEST_ID --yes
```

A project that was already created is recovered immediately. Otherwise, give
the same name, repository and branch to resume. Reusing an ID with different
settings is refused.

### Looking at a project

```sh
parlant dashboard project list
parlant dashboard project describe PROJECT_ID
parlant dashboard agent list
parlant dashboard agent view AGENT_ID
```

### Deploying

A managed project deploys one attempt at a time. `deploy` builds the configured
branch's latest commit, or exactly the commit you name, with the settings and
secrets saved at that moment. It then follows the attempt until it is live or
has failed.

```sh
parlant dashboard deploy
parlant dashboard deploy --commit COMMIT_SHA
```

`--commit` takes a full 40-character SHA. The command stops waiting after
14 minutes, and Ctrl-C stops only the watching — neither cancels anything, and
the deployment carries on. If another attempt or a repository check is already
running, `deploy` names it and exits rather than starting a second one.

```sh
parlant dashboard deployment list
parlant dashboard deployment view ATTEMPT_ID
parlant dashboard deployment logs ATTEMPT_ID --channel build --follow
parlant dashboard deployment stop ATTEMPT_ID
```

`deployment list` shows attempts newest first, marks the one serving traffic
`(live)`, and prints a `--cursor` command when there are more pages.

`deployment logs` reads one channel of an attempt's retained output: `build`
(the default), `deployment` (starting up and going live), or `runtime` (the
live app). Without `--follow` it prints what is retained now. If earlier output
has fallen out of retention it says how many lines are gone; if output falls
out while you are following, it says so and continues with what is still there,
without repeating lines it already printed.

`deployment stop` asks for confirmation, which `--yes` skips. A stop is
acknowledged first and is final only once the platform confirms it — check with
`deployment view`. Stopping an attempt never takes down the deployment that is
already live.

There are no separate `build` or `deploy` groups to inspect: one deployment
attempt covers both building and going live, and `deployment` is how you look
at it.

### Secrets

Secrets are part of a project's saved settings.

```sh
parlant dashboard secret list
parlant dashboard secret set STRIPE_API_KEY=value
parlant dashboard secret unset STRIPE_API_KEY
```

`secret list` shows names only; values are never returned. To keep a value out
of your shell history and the process list, give the key alone and pipe the
value in:

```sh
printf %s "$STRIPE_API_KEY" | parlant dashboard secret set STRIPE_API_KEY
```

Set secrets yourself rather than sending their values to anyone. Values are
limited to 64 KiB. A running deployment keeps the secrets it started with, so
run `parlant dashboard deploy` to apply a change.

### Reading sessions

```sh
parlant dashboard session list
parlant dashboard session list --agent AGENT_ID --limit 20
parlant dashboard session view SESSION_ID
parlant dashboard session summary SESSION_ID
parlant dashboard session events SESSION_ID --kind tool
```

`session view` shows a session as turns. `session events` shows the raw event
stream, which is what you want when you care about tool calls or ordering.

When an agent did something you did not expect, `session inspect` shows why
that turn came out the way it did:

```sh
parlant dashboard session inspect SESSION_ID --turn TURN_ID
parlant dashboard session inspect SESSION_ID --turn TURN_ID --include-logs
```

Engine traces carry the same turn from the engine's side:

```sh
parlant dashboard trace view TRACE_ID
```

### Usage and logs

```sh
parlant dashboard analytics view --range last-7-days
parlant dashboard analytics labels --range last-30-days
parlant dashboard logs
```

`logs` follows a project's runtime logs and runs until you stop it. A running
project keeps producing logs, so there is no completion to wait for.


## Output for programs

Some commands take `--json` and then print only their result on stdout, with
everything else on stderr. These do:

```
parlant init
parlant docs query
parlant dashboard auth login | logout | status
parlant dashboard workspace list | use
parlant dashboard project check | confirm | connect | create | list
```

`--json` never prompts, so it needs the answers up front: `parlant init --json`
requires `--package-manager`, and `--yes` as well when creating a dashboard
project.

Failures print as `{"error": {"code": ..., "message": ...}}` and keep their
exit code, so a script can tell `NOT_SIGNED_IN` from `FILES_EXIST` without
reading English.

The other commands print tables for people to read.


## Workspace CLI allowances

Some dashboard reads are pooled across a workspace's members, credentials and
projects. There are three allowances: trace reads, log reads, and inspections.

Inspections cover session-turn reads and the starting of a managed setup check.
Polling a check that is already running does not spend another unit, and the
check that starts on its own when a project is created is not counted against
you. Sessions in the browser, and Parlant's own internal calls, do not spend
these allowances at all.

The commands that spend them are:

```sh
parlant dashboard trace view TRACE_ID
parlant dashboard session inspect SESSION_ID --turn TURN_ID
parlant dashboard logs
parlant dashboard deployment logs ATTEMPT_ID
```

When an allowance runs out, the command tells you which one, the limit, how
much is left, and when it resets (or `unknown`, if that is not supplied). It
exits with status **7**.

It does not sleep, retry, reconnect, or forget your sign-in because of a
refusal. Following logs can also reach a time limit for a single stream: the
command stops with status **7** and tells you to start a new log request.
Opening another stream is a new request and is allowed.

Limits, and how long a stream may run, are part of your plan. The CLI holds no
limits of its own, so what it reports is whatever your workspace actually has.


## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | The command failed; the message says why |
| 2 | The command line was wrong (bad flag, missing argument) |
| 7 | A workspace CLI allowance or log-stream limit was reached |
