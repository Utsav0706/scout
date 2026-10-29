# From GLITCH to Scout: state machines, utility scores and an LLM agent

**GLITCH** was my MSc project: a Unity third-person shooter whose world "glitches"
(gravity flips, walls vanish, time stutters, enemies and the player get duplicated). I
built two enemy brains for it, a **finite state machine** and a **Utility AI**, and
compared them on 195 enemies each. **Scout** is an LLM research agent built as a
LangGraph state graph. Different domains, but the same design questions: what state is
the agent in, what moves it to the next one, and how does it choose between actions?

## The same ideas, side by side

| Idea | GLITCH | Scout |
|---|---|---|
| A unit of behaviour | `IState` with `Enter / Tick / Exit` (Patrol, Chase, Attack, TakeCover, Frozen) | A graph node: an async function from state to a state update (planner, researcher, reader, writer, verifier, reviser, finaliser) |
| Moving between them | One `Evaluate()` runs priority-ordered rules every frame | Edges, plus two conditional edges (`fan_out`, `after_reviser`): pure functions of the state |
| Hard stops first | Dead, then frozen, are checked before anything else | `RunBudget.check()` (steps, £, seconds) runs before every model call |
| Retreat when weak | Below a health threshold, take cover | Below the finish reserve, stop researching and write with what you have |
| Doing several things at once | Not possible: one state at a time (a weakness my design report noted for overlapping glitches) | `Send` fans out one researcher per sub-question, in parallel |
| Choosing an action | `UtilityBrain` scores every action with explicit considerations (e.g. `hurt × exposure + glitch bonus`) and picks the best, every 0.25 s | The researcher model chooses which tool to call; the "scoring" happens inside the model |
| Seeing why it chose | Logged the top and second-best score, plus a debug overlay | Every model and tool call traced to Langfuse and stored as a trajectory |

## What carried over

**1. Rules for what must be predictable; scoring or a model for what can't be listed.**
- GLITCH's FSM kept life-or-death checks as fixed rules, and left the open choices
  (cover or reposition? chase or spread fire?) to utility scores.
- Scout does the same:
  - budgets, the single retry round, tool allow-lists and link allowlists are plain
    code that can't be talked out of its job
  - only the open-ended choices (what to search, what a source says, how to phrase a
    claim) go to the model
- A prompt-injected web page can change what the model *wants* to do, but not what the
  graph *lets* it do.

**2. Utility scores are transparent; an LLM's choices are not, so measure them.**
- In GLITCH I could read every consideration and tune a curve. A model's tool choice is
  more flexible (it handles any question without a hand-written consideration for each)
  but opaque and not repeatable.
- So I treat it the way I treated the utility brain in the dissertation:
  - **log the decision** (GLITCH logged winning and runner-up scores to show decisions
    were really scored rather than defaulted; Scout logs every tool call)
  - **test it against a baseline**: Scout's golden set checks tool selection on every
    case (50/50 correct in the baseline)
- Scout also narrows the choice, the way utility scores are clamped: the researcher is
  only offered the tools the user enabled, and refuses any other.

**3. Report the half that wasn't supported.**
- GLITCH's result was half a win:
  - Utility enemies used a wider repertoire (4.10 vs 3.80 distinct behaviours,
    p = 0.004, Mann–Whitney U)
  - but they were *not* less predictable (action entropy 1.34 vs 1.32, p = 0.235)
  - and they died slightly more often
- Scout's baseline has the same shape:
  - strong on document questions (12/12) and on refusing to invent answers (8/8)
  - weak on questions needing both web and documents (1/10)
  - its own verifier passes every kept claim, while an independent judge agrees with
    66% of citations
- Both projects say so in their results instead of leading only with the good number.

## Where the analogy breaks

- **A game AI reacts forever; Scout runs an episode.** An enemy's state machine
  re-evaluates every frame for as long as the game runs. Scout's graph runs once from
  question to report, with one bounded loop. That makes a hard budget possible, and
  necessary, because every step costs money.
- **Errors look different.** A bad utility score gives visibly bad behaviour on screen.
  A bad model step can give *plausible* text. That's why Scout checks quotes word for
  word in code rather than trusting the verifier's verdict.

## In short

- **The same split:** fixed rules for what must be predictable (budgets, the retry
  limit, allow-lists) wrap the part that makes open-ended choices. In GLITCH that part
  was a utility scorer; in Scout it's a language model.
- **Logged decisions:** every choice is logged so it can be inspected, through a debug
  overlay and score logs in GLITCH and through traces and trajectories in Scout.
- **Measured against a baseline:** utility scores can be read directly, but a model's
  choices can't, so Scout's are measured with a 50-question test set. GLITCH's were
  measured on 390 enemies.
- **Weak spots reported:** both projects report where they fall short, alongside where
  they do well.
