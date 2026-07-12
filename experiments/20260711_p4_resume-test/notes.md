# 20260711_p4_resume-test

**Hypothesis:** killing a run mid-training (real `kill -INT`, i.e. Ctrl-C) and resuming via
`scripts/train.py --resume` reproduces the exact per-step losses an uninterrupted run would
have produced -- the phase-4 exit criterion "resume verified," and the payoff of loader.py's
stateless `(seed, step)` sampling design.

**Observation (two bugs found and fixed along the way, not just a clean pass):**
1. `wandb.init()` silently installs its own SIGINT handler, so a plain `kill -INT` on the
   trainer process was being swallowed entirely -- `except KeyboardInterrupt` never fired.
   Fixed by reinstalling `signal.default_int_handler` right after `wandb.init()`
   (`trainer.py`).
2. Once Ctrl-C actually worked, the *first* resume attempt exposed a real off-by-one: the
   checkpoint recorded the *just-completed* step instead of the *next* one to run (the
   for-loop's own iteration variable was doing double duty as both "step being executed" and
   "step to checkpoint"). Resuming re-executed the already-completed step, re-applying its
   gradient update on top of a model that had already taken it -- caught because the replayed
   step's logged loss (9.400) didn't match the original run's (9.706), which is only possible if
   the model had already moved. Fixed by decoupling the loop's local `step` variable from
   `self.step` (now only bumped to `step + 1` after all of that step's eval/log/sample work
   completes).

**Conclusion:** after both fixes, killing mid-run and resuming reproduces every subsequent
logged loss bit-for-bit against an uninterrupted control run (steps 0,2,4,5,6,8 all matched to
the last decimal against `20260711_p4_cpu-canary`'s original numbers). Resume is now genuinely
bit-exact, not just "close." Neither bug would have been caught by the unit test alone
(`tests/test_trainer.py`'s resume test manages `step` correctly by construction) -- this run is
why CLAUDE.md's "verify on a real run" instinct matters even when the unit tests are green.
