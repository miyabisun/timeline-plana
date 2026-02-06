# Tactical Guidance

## Purpose

The core mission of Timeline Plana. Reads a YAML timeline file (created by **Timeline Arona**) and, using real-time game state, guides the Sensei (Player) on **which EX skill to activate next** and **when**.

This is the feature that transforms raw screen data into actionable combat advice.

## Timeline File

A YAML file authored in **Timeline Arona** that encodes a boss-fight strategy as a sequence of skill-activation rules. Each rule defines a **trigger condition** and a **target skill**:

| Trigger Type | Example | Description |
|---|---|---|
| Time-based | `at: "1:30"` | Activate the specified skill when the countdown timer reaches the given time. |
| Cost-based | `when_cost: 3` | Activate the specified skill when the accumulated EX skill cost reaches the threshold. |

Rules are evaluated in order. The next rule whose trigger has not yet been satisfied becomes the **current guidance target** displayed to the Sensei.

## Core Loop

1. **Load** — Parse the YAML timeline file into an ordered list of activation rules.
2. **Monitor** — Continuously read game state via backend sensors:
   - **Remaining Time** — supplied by `countdown_monitor`.
   - **EX Skill Cost** — read from the game screen via OCR *(planned)*.
   - **Battle State** — supplied by `combat_intel`. Guidance is paused during non-Active states.
3. **Resolve** — Compare current game state against the timeline rules. Determine the **next skill to activate** and its **target timing**, and advance the pointer once a rule's trigger condition is satisfied.
4. **Display** — Push the next-action payload to the Frontend via `shittim_link`. The UI renders the guidance on the Sensei's screen, enabling smooth, timing-accurate skill activation.

## Integration Points

| System | Role in Tactical Guidance |
|---|---|
| `countdown_monitor` | Supplies real-time remaining time for time-based trigger evaluation. |
| `combat_intel` | Supplies battle state; halts guidance advancement during Paused / Slow / Inactive. |
| `shittim_link` | Delivers the current next-action payload (skill name, target time) to the Frontend at 30 Hz. |
| Frontend UI | Renders the skill activation prompt for the Sensei in real time. |
