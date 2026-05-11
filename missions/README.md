# Missions

Missions are YAML files that describe what the agentic test framework should do. Each file
contains a `missions:` list; each entry is a single isolated test run.

## Schema

```yaml
missions:
  - thread_id: "<unique_id>"   # required — also routes to a graph (see below)
    prompt: >                  # required — natural-language instructions for the agent
      <multi-line text>
```

### `thread_id`

* Must be unique per run. It keys the persistent SQLite checkpoint and the artifact
  directory (`report_<thread_id>/`).
* Reusing the same `thread_id` resumes the prior conversation. Pass `--clear-memory` to
  start fresh.
* **Routing keywords**: if `thread_id` contains any of `accessibility`, `a11y`,
  `data_heavy`, `data-heavy`, `impatient`, `returning`, `explorer`, `chaos`, or
  `autonomous`, the mission is dispatched to the **advanced** graph. Otherwise it
  runs on the **standard** 3-persona swarm.

### `prompt`

* Free-form natural language. The supervisor reads it and decides which specialist agent
  should drive the test.
* Mention the user persona or risk area you want exercised so the supervisor routes to the
  right agent.
* Use placeholders for app-specific values — for example `<YOUR_APP>`, `<APP_URL>`,
  `<example_search_term>`, `<dashboard_path>` — and replace them before running.

## Standard agents

| Agent                      | Specialization                                                        |
|----------------------------|-----------------------------------------------------------------------|
| `new_user_agent`           | Tests onboarding flows, discoverability, default states, empty states |
| `power_user_agent`         | Keyboard shortcuts, bulk operations, advanced filters, edge cases     |
| `adversarial_user_agent`   | Deliberately breaks things (malformed inputs, back-button abuse)      |

## Advanced agents

| Agent                      | Specialization                                                        |
|----------------------------|-----------------------------------------------------------------------|
| `impatient_user_agent`     | Rapid interactions, cancels operations, refreshes during load         |
| `accessibility_user_agent` | Validates WCAG, screen reader nav, keyboard-only interaction          |
| `data_heavy_user_agent`    | Uploads large files, thousands of records, excessively long strings   |
| `returning_user_agent`     | Stale sessions, cached pages, outdated bookmarks                      |
| `explorer_agent`           | Autonomous chaos exploration; finds crashes, timeouts, regressions    |

## Writing a new mission

1. Pick a `thread_id` that signals the persona or area under test (e.g. `smoke_new_user_01`).
2. Write a prompt that tells the agent **what to do** and **what to verify**, not how.
3. Reference the persona or risk area the supervisor should route to.
4. If the test should run on an advanced agent, include one of the advanced routing keywords
   in the `thread_id`; for autonomous exploration, use `explorer`, `chaos`, or `autonomous`.

Each supported agent has a generic mission template named `<agent>.yaml` in this folder to help you get started.
