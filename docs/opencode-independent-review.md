# Independent OpenCode review using dedicated subagent roles

Scope: TSUMGI change review and acceptance gates; first validated against PR #123
on 2026-10-10 JST. This workflow is **read-only**. It does not commit,
approve a GitHub PR, waive a human-gated policy, or deploy anything.

## Selection rule

On the authorized OpenCode host, the installed `reviewer` and
`security-reviewer` are **subagents**, not primary agents. The command
`opencode run --agent reviewer` falls back to the default primary agent
and **must not** be counted as a Reviewer-role run.

Use the **primary** `plan` agent in a separate, read-only tmux session.
`plan` must call its `task` tool with the exact
`subagent_type=reviewer` and `subagent_type=security-reviewer`.
Do not count a primary model's self-written review, a described intent to
delegate, or a running task as an independent reviewer result.

For nontrivial design, use `architect` as another scoped read-only
subagent, but do not allow the Reviewer to change code or issue GitHub
mutations.

## Exact-head and permission preflight

1. Resolve and record the exact PR head commit, base commit, and reviewed
   file/diff set through trusted GitHub read-only metadata. Clone the
   exact revision to a **new isolated directory** (mode 0700);
   ensure `git status --porcelain` is empty. Preserve existing
   OpenCode/tmux sessions and every production checkout.
2. Confirm both agents are installed using `opencode agent list` and
   their role metadata explicitly says `mode: subagent`. Check that the
   intended models are available and that the primary Agent has
   `task: reviewer` and `task: security-reviewer` permissions.
3. Create a **temporary read-only OpenCode config** outside the repository:
   default deny; allow only `read`, `glob`, `grep`, `list`; allow
   `plan` to call the two named subagents through `task`. Deny
   `edit`, `apply_patch`, `bash`, external directories, network,
   `question`, and other task targets for all reviewer children.
   Do not change global OpenCode auth or shared configuration.
4. Prefer `reviewer` reasoning effort **High** (rather than Max) for
   bounded experiments; measure the effective model, child identity,
   output and timeouts. The observed setup used a primary
   GPT-6.1 Sol fast model, `reviewer` on GPT-6 Luna with a temporary
   High setting, and `security-reviewer` on GPT-5.6 Terra. These
   are verification examples, not a permanent model lock.
5. On affected environments, launch OpenCode **inside its own tmux**
   socket/session; previous headless non-TTY calls hung with no output.
   Use a concrete time budget, preserve JSON events and stderr in
   mode-0600 files, and never claim a timed-out call was reviewed.

## Independent review contract

Explicitly ask the primary to invoke the actual `task` tool, first with
`subagent_type=reviewer` and separately with
`subagent_type=security-reviewer`. Review the exact frozen source and the
full changed diff, surrounding contracts, tests and historical E2E
limitations. Require each child to return:

- `status: COMPLETED` or a concrete blocking status;
- inspected files, precise line-level findings with severity, conditions,
  impact and minimal remedial action, distinguishing confirmed from
  speculative;
- clear `No findings` only if the inspected scope and evidence justify it;
- unexecuted test and runtime verification as explicit **NOT_RUN**, never PASS.

The parent collates **both actual child outputs**, preserving differences
and any unknowns. A prompt asking for only one finding proves the wiring
but **does not satisfy a complete independent review gate**.

## Evidence required to count a role run

- Primary OpenCode JSON transcript includes a `task` invocation with
  `subagent_type=reviewer` or `security-reviewer` and completed output.
- Independently export the **child session** using
  `opencode export --sanitize <child-session-id>`. Verify the child's
  recorded agent name is the requested role and its model identity,
  and that it has a completed final message. A Subagent CLI warning
  about fallback **invalidates** a direct-role claim.
- Retain sanitized public-safe findings and SHA-256 commitments to
  private mode-0600 traces. Do not publish OAuth, tokens, raw private
  model outputs, infrastructure addressing or user prompts.
- Independently check the reviewed PR head **after** completion. Any
  code change since the reviewed revision requires reviewing the delta
  or repeating the bounded review and the relevant regressions.

The child reports are **AI code review evidence**, not independent human
approval, not a GitHub `APPROVE` action, not a merge grant, and not a
production release gate.

## Evidence from PR #123 wiring validation

On one pinned read-only clone, both real role sessions were observed:

- `reviewer` / GPT-6 Luna returned `status: COMPLETED` and a P2
  finding: a child cleanup exception can mask original startup
  failure/cancellation in the composite Runtime.
- `security-reviewer` / GPT-5.6 Terra returned
  `status: COMPLETED` and a conditional medium finding: with
  mutable model storage, the pre/post GGUF hash does not prove which
  inode the subprocess loaded between checks.

A direct `--agent reviewer` CLI invocation warned and fell back to the
default primary agent. An unrestricted high-effort invocation previously
timed out; only the confirmed `task`-mediated role sessions count.
Each diagnostic asked for one finding, so the remaining full-diff
review is still required after any remediation.

See PR #123 and issue #122 for exact-head tests and immutable GPU evidence.

## Integration gate

After the changed implementation passes tests and actual GPU E2E,
re-request **unrestricted scoped correctness + security** reviews at the
updated exact HEAD. Unresolved blocking findings mean HOLD. A green
CI pipeline or a completed role invocation alone cannot waive approval.
Merging the feature PR, integrating main, and reversible
production Control/Worker cutover each require their **separate** recorded
authority and relevant release gates.
