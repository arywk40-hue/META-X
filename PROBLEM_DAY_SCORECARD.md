# Problem Day Scorecard

Use this document the moment the official problem statement drops.

The goal is simple:

- clear the validator and deployment gates
- maximize the weighted judging criteria
- avoid spending time on low-value polish before the high-weight areas are strong

## Pass/Fail Gates First

If any of these fail, the rest of the score does not matter much:

- HF Space deploys and responds
- Dockerfile builds successfully
- `openenv.yaml` is valid and coherent
- `/health`, `/reset`, `/step`, `/state`, `/tasks`, `/grader` behave correctly
- root [inference.py](/Users/ariyanbhakat/Desktop/metax/inference.py) exists and runs without crashing
- at least 3 tasks exist
- graders are deterministic and return scores in `[0.0, 1.0]`

## Weighted Judging Criteria

| Criterion | Weight | What judges will care about most |
|---|---:|---|
| Real-world utility | 30 | Does this model a genuine human task with real evaluation value? |
| Task & grader quality | 25 | Are tasks clear, meaningful, and fairly graded with real progression? |
| Environment design | 20 | Are the state, actions, rewards, and episode boundaries sensible? |
| Code quality & spec compliance | 15 | Does it pass the OpenEnv bar cleanly and look production-ready? |
| Creativity & novelty | 10 | Does it feel original and interesting, not just another generic benchmark? |

## Scoring Worksheet

Fill this in when comparing candidate domain interpretations.

### 1. Real-World Utility (30)

Score bands:

- `0-5`: toy task, artificial puzzle, little practical value
- `6-15`: real-ish domain, but shallow or weakly modeled
- `16-25`: useful agent evaluation task with believable workflow
- `26-30`: immediate value to the RL / agent community

Questions:

- Is this something a real human actually does?
- Would someone use this environment to compare agents in practice?
- Are the observations realistic enough to support genuine reasoning?
- Does success in the environment translate to real capability?

Our score:

- Candidate domain:
- Utility score:
- Why:

### 2. Task & Grader Quality (25)

Questions:

- Do we have at least 3 tasks with meaningful easy / medium / hard progression?
- Are the task objectives explicit and unambiguous?
- Do graders reflect true success, not superficial shortcuts?
- Are graders deterministic and reproducible?
- Does the hard task actually challenge strong models?

Red flags:

- hard task is just more rows / more tokens instead of harder reasoning
- grader can be gamed by format-only behavior
- grader always returns similar values
- task descriptions are vague

Our score:

- Task quality score:
- Grader quality score:
- Main risks:

### 3. Environment Design (20)

Questions:

- Does `reset()` always produce a clean, valid starting state?
- Are `Observation` and `Action` fields compact, typed, and useful?
- Are the available actions explicit enough to constrain agent behavior?
- Does the reward function provide useful step-level signal?
- Are episode boundaries sensible and resistant to looping or degenerate behavior?

Red flags:

- sparse reward only at the end
- confusing or bloated observation payloads
- actions too unconstrained to grade cleanly
- no penalty for wasting steps or invalid moves

Our score:

- Design score:
- Main risks:

### 4. Code Quality & Spec Compliance (15)

Questions:

- Does `python scripts/validate.py` pass?
- Do tests pass?
- Does Docker build?
- Does the HF Space respond?
- Is the repo clean, typed, documented, and easy to inspect?

Red flags:

- validator passes locally but inference is untested
- manifest and README disagree
- fragile runtime assumptions
- undocumented environment variables

Our score:

- Compliance score:
- Main risks:

### 5. Creativity & Novelty (10)

Questions:

- Is this domain less common than the obvious generic choices?
- Is there an interesting mechanic in the reward or grading?
- Does the environment expose a subtle but meaningful capability gap?

Red flags:

- same benchmark idea everyone will ship
- novelty only in wording, not in mechanics
- “creative” features that weaken realism

Our score:

- Novelty score:
- Why:

## Build Priorities By Expected Judge Impact

When time is tight, prioritize in this order:

1. Make the domain feel genuinely real
2. Make the graders fair, deterministic, and hard to game
3. Make the easy / medium / hard progression meaningful
4. Make reward shaping informative and stable
5. Keep the validator and inference path clean
6. Add novelty only if it does not weaken the first five

## Domain Selection Filter

Before we commit to a domain interpretation, it should pass all of these:

- It is not a game or toy puzzle
- It can be explained in one sentence as a real human workflow
- It supports at least 3 difficulty levels naturally
- It can be graded programmatically without hidden human judgment
- It gives partial credit in a believable way
- A baseline LLM can attempt it from structured observations

If any of these fail, keep looking.

## Launch-Day Execution Plan

1. Read the official statement and list 2-3 possible domain framings.
2. Score each framing quickly with this document.
3. Choose the framing with the highest combined score on:
   - real-world utility
   - task & grader quality
4. Implement the domain swap in:
   - [environment/models.py](/Users/ariyanbhakat/Desktop/metax/environment/models.py)
   - [environment/tasks.py](/Users/ariyanbhakat/Desktop/metax/environment/tasks.py)
   - [environment/graders.py](/Users/ariyanbhakat/Desktop/metax/environment/graders.py)
   - [environment/reward.py](/Users/ariyanbhakat/Desktop/metax/environment/reward.py)
5. Update:
   - [openenv.yaml](/Users/ariyanbhakat/Desktop/metax/openenv.yaml)
   - [README.md](/Users/ariyanbhakat/Desktop/metax/README.md)
   - [inference.py](/Users/ariyanbhakat/Desktop/metax/inference.py) if prompting/action extraction must change
6. Run:
   - `python -m pytest -q`
   - `python scripts/validate.py`
7. Re-score the final build with this scorecard before submission.

## Final Submission Sanity Check

Before we submit, we should be able to say yes to all of these:

- This is a real task, not a toy
- Easy / medium / hard are genuinely different
- The hard task can separate frontier models from weak ones
- The reward gives useful signal before the end of the episode
- The grader is deterministic and cannot be trivially exploited
- The Space deploys and inference runs end-to-end

If we cannot say yes to those six, the submission is not done.
