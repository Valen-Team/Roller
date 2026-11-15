# Security policy

## Supported versions

Fixes are released for the latest version of `roller-agent` on PyPI.

## Reporting a vulnerability

Report a vulnerability privately through GitHub: open the [Security tab](https://github.com/Valen-Team/Roller/security) of this repository and choose **Report a vulnerability**. Include the Roller version, the Harness and image involved, and steps to reproduce.

Do not open a public issue for a vulnerability.

## Scope

Roller runs agents in Docker containers on the host and passes API keys to them through environment variables. Reports of interest include a task or model output that reaches host files outside the declared inputs and outputs, credentials written to a run directory or log, and container options that weaken isolation by default.

Trajectories, wire traces and `capture=full` Context contain the prompts, tool arguments and results of a run. Treat a run directory as sensitive data; that content being present in it is the documented behaviour.
