# Documentation

- [`plan.md`](plan.md): the main plan: teams, where teams meet, checkpoints, demo, safety.
- [`map.html`](map.html): one-page visual map (architecture, what happens when an arm is plugged in,
  speed-up limits, who builds what). GitHub shows it as source: download it and open it in a browser.
- [`../tasks/README.md`](../tasks/README.md): how to add a task.
- [`../research/`](../research/): background report and notes.

## How the parts connect

```mermaid
flowchart LR
    T["task folder<br/>steps, order, shared resources"] --> G["taskgraph<br/>load and validate"]
    G --> S["scheduler<br/>who does what, when<br/>+ lower bound"]
    S -- plan --> E["executor<br/>run, re-split on change"]
    E -- re-plan --> S
    E -- "execute(task)" --> W["arm workers<br/>real / sim / fake"]
    W -- Result --> E
    D["discovery<br/>arm plugged in / out"] -- arm events --> E
    E --> L["event log"]
    L --> V["live screen<br/>Gantt, finish time, bottleneck"]

    classDef ops fill:#e6f2ec,stroke:#2e7d4f,color:#16212b
    classDef act fill:#e7f0fb,stroke:#1d76db,color:#16212b
    classDef hw fill:#fbeaea,stroke:#b60205,color:#16212b
    class G,S,E,L,V ops
    class W act
    class D hw
```

Colours: green Operations (#21), blue Action (#20), red Hardware (#19). Task folders are written by whoever
builds that task.

## When an arm is plugged in

1. Discovery sees a known arm and starts its worker.
2. The worker checks the arm's state and limits, moves slowly to home, and reports ready.
3. The executor keeps running tasks where they are and re-plans the rest.
4. The screen shows the new finish time and what now limits it (arm count, step order, or a shared resource).
