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
* **Routing keywords**: if `thread_id` contains any of `explorer`, `chaos`, or
  `autonomous`, the mission is dispatched to the **advanced** graph (autonomous
  exploration). Otherwise it runs on the **standard** 5-agent UI swarm.

### `prompt`

* Free-form natural language. The supervisor reads it and decides which specialist agent
  should drive the test.
* Mention the UI pattern you want exercised (lists, charts, maps, forms, graphs) so the
  supervisor routes to the right specialist.
* Use placeholders for app-specific values — for example `<YOUR_APP>`, `<APP_URL>`,
  `<example_search_term>`, `<dashboard_path>` — and replace them before running.

## Standard agents (UI-pattern specialists)

| Agent             | Specialization                                                        |
|-------------------|-----------------------------------------------------------------------|
| `listing_agent`   | Lists, tables, search bars, filters, pagination, row detail flyouts   |
| `graph_agent`     | Node-link graphs, timelines, waterfalls, hierarchical visualizations  |
| `chart_agent`     | Time-series, bar/line/area charts, KPI tiles, dashboards              |
| `map_agent`       | Geographic maps, status grids, geospatial overlays                    |
| `form_agent`      | Forms, wizards, validation, multi-step configuration                  |

## Generic Exploration Personas (Behavioral strategies)

| Agent                      | Specialization                                                        |
|----------------------------|-----------------------------------------------------------------------|
| `new_user_agent`           | Tests onboarding flows, discoverability, default states, empty states |
| `power_user_agent`         | Keyboards shortcuts, bulk operations, advanced filters, edge cases    |
| `adversarial_user_agent`   | Deliberately breaks things (malformed inputs, back-button abuse)      |
| `impatient_user_agent`     | Rapid interactions, cancels operations, refreshes during load         |
| `accessibility_user_agent` | Validates WCAG, screen reader nav, keyboard-only interaction          |
| `constrained_user_agent`   | Tests degraded paths, slow networks, small viewports                  |
| `data_heavy_user_agent`    | Uploads large files, thousands of records, excessively long strings   |
| `returning_user_agent`     | Stale sessions, cached pages, outdated bookmarks                      |

## Advanced agents

| Agent             | Mission                                                               |
|-------------------|-----------------------------------------------------------------------|
| `explorer_agent`  | Autonomous chaos exploration; finds crashes, timeouts, regressions    |

## Writing a new mission

1. Pick a `thread_id` that signals the area under test (e.g. `smoke_listing_users_01`).
2. Write a prompt that tells the agent **what to do** and **what to verify**, not how.
3. Reference the UI patterns the supervisor should route to.
4. If the test should run autonomously without scripted steps, name it with `explorer`,
   `chaos`, or `autonomous` to dispatch to the advanced graph.

Each persona has a generic mission template named `<persona>_agent.yaml` in this folder to help you get started.
