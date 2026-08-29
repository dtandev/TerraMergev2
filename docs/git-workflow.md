# Git, CI & AI-Agent Workflow — TerraEye S&I

Team-wide rules for version control, mandatory checks, and AI coding assistants (Claude
Code or any other agent working in these repos). Applies to every project created from
this template. If a project's checks or hooks drift from what's described here, that's a
bug in the project — bring it back in line with the template, don't rewrite this doc to
match the drift.

---

## 1. Branch model

- `main` is always releasable. Nobody — human or AI agent — commits or pushes to `main`
  directly, including "quick fixes" and docs typos.
- All work happens on a feature branch: `feat/...`, `fix/...`, `chore/...`, `docs/...`
  (prefix matches the Conventional Commit type you'll use).
- The only way into `main` is a Pull Request that has been reviewed and has green CI.
- If you don't know what branch is currently checked out, run `git status` / `git branch
  --show-current` before doing anything else — don't guess.

## 2. Commit messages & PR titles — Conventional Commits

Every commit message and every PR title must follow [Conventional
Commits](https://www.conventionalcommits.org/):

```
feat: add cloud mask filtering      # new feature  → MINOR bump
fix: handle empty AOI input         # bug fix       → PATCH bump
feat!: redesign CLI interface       # breaking      → MAJOR bump
chore: update dependencies          # housekeeping  → no bump
docs: clarify AOI input format
```

This isn't just style — `cz bump` reads commit history to compute the next version and
generate the changelog, and it breaks silently if commits don't conform.

Enforced twice, on purpose:
- **Locally**, at commit time: the `commitizen` pre-commit hook (`commit-msg` stage)
  rejects a bad commit message before it's even created.
- **On GitHub**, at PR time: `semantic-pull-request-check.yaml` rejects a non-conforming
  PR title. This exists because squash-merge uses the PR title as the commit message on
  `main` — a good commit history but a sloppy PR title still pollutes `main`.

## 3. Pre-commit hooks are mandatory, not optional

`just setup` installs them for you (`pre-commit install` + `pre-commit install --hook-type
commit-msg`). Do this once per clone. The hook set (`.pre-commit-config.yaml`):

| Hook | Purpose |
|---|---|
| `ruff` + `ruff-format` | lint & format Python, autofix on commit |
| `trailing-whitespace`, `end-of-file-fixer` | file hygiene |
| `check-yaml`, `check-toml`, `check-json` | catch broken config files before they merge |
| `check-added-large-files` (5MB cap) | keep the repo out of git-lfs territory by accident |
| `check-merge-conflict` | catch leftover `<<<<<<<` markers |
| `debug-statements` | catch stray `pdb.set_trace()` / `print` debugging leftovers |
| `commitizen` | enforce Conventional Commits (see §2) |
| `detect-secrets` | block commits containing API keys/tokens/passwords |

**Never run `git commit --no-verify` (or otherwise skip hooks) to "get past" a failing
check.** A failing hook means something is genuinely wrong — fix it, don't silence it. If
a hook is a false positive, fix the hook config (e.g. update `.secrets.baseline`) in the
same PR, with a reason in the commit message — don't bypass it ad hoc.

### Why local hooks aren't enough on their own

Local hooks only run if the contributor has them installed and doesn't skip them.
`pre-commit.yaml` closes that gap: it re-runs the **entire** pre-commit suite in CI on
every push and PR, so a missing local install or a `--no-verify` never silently reaches
`main` unnoticed. Treat a red check here the same as a failing test — it blocks merge.

## 4. CI checks — what runs and why

| Workflow | Trigger | Blocks merge on |
|---|---|---|
| `pre-commit.yaml` | every push, every PR | any pre-commit hook failing (§3) |
| `check.yml` ("Baseline Check") | PR → `main`/`master` | ruff lint+format, `pytest`, required baseline files/dirs present |
| `semantic-pull-request-check.yaml` | PR opened/edited/synced | PR title not a valid Conventional Commit |
| `claude-pr-review.yaml` | PR opened/synced/reopened | (advisory) posts an automated Claude code review comment — doesn't block, but read it |
| `release.yml` | push to `main` (i.e. after a PR merges) | not a PR check — see §6 |

**Don't edit or disable a workflow file to make a PR go green.** If a check is
genuinely wrong for a change, that's a discussion for the PR review, not a unilateral
edit to `.github/workflows/`. Treat changes to `.github/workflows/**` or
`.pre-commit-config.yaml` themselves as higher-scrutiny — call them out explicitly in the
PR description.

### The enforcement gap: branch protection

CI producing a red ❌ does **not**, by itself, stop a merge or a direct push — GitHub only
enforces that if branch protection rules are turned on for `main`. As of writing, this
repo/template has **no branch protection configured**, and pushing to `main` requires
only standard `push` permission (most contributors have it). That means today, the
"never touch `main`" rule is a matter of discipline (human and AI), not a platform
guarantee.

**Action for whoever has admin on the GitHub org/repo:** enable, on `main`, for every
repo created from this template:
- Require a pull request before merging (no direct pushes, including for admins if
  possible).
- Require status checks to pass before merging — select the `check` job, `pre-commit`
  job, and the semantic PR title check by name.
- Require branches to be up to date before merging.
- Disallow force-pushes and branch deletion on `main`.
- (Recommended) Require at least one approving review.

Until that's enabled org-wide, treat §1 and §7 below as the load-bearing safeguard.

## 5. Secrets & security

- Real secrets live in `.env`, which is git-ignored. Only `.env-example` (placeholder
  values) is committed.
- `detect-secrets` scans every commit against `.secrets.baseline`. If it flags a false
  positive, regenerate the baseline (`detect-secrets scan > .secrets.baseline`) and say
  why in the PR — don't just delete the finding.
- If a real secret is ever committed (even briefly, even on a branch that got deleted):
  rotate the credential immediately. Removing it from a later commit does not remove it
  from git history.

## 6. Versioning & releases

Versioning is [SemVer](https://semver.org/), computed by `commitizen` from commit
history — but **never on a feature branch**. Two PRs open at the same time would each
compute "the next version" from their own, possibly-stale view of `main`, and could both
land on the same number — a silent collision with no platform-level guard against it.

Instead, releasing is fully automatic and happens strictly *after* a PR merges:

- A feature branch / PR **never touches `pyproject.toml`'s version, `product.yaml`'s
  version, `uv.lock`, or `CHANGELOG.md`**. The only thing a PR is responsible for is a
  Conventional-Commit-compliant title (§2) — that's the sole input the release process
  reads.
- On every push to `main` (i.e. every merge), `release.yml` checks out the *actual,
  current* `main`, runs `cz bump --changelog` to compute the next version from commits
  since the last tag (MINOR for `feat`, PATCH for `fix`, MAJOR for a breaking change),
  commits the version bump + changelog entry directly to `main`, and pushes the matching
  `vX.Y.Z` tag.
- Two commitizen settings make this actually work end to end: `version_provider =
  "pep621"` (edits `pyproject.toml`'s `[project].version` via a real TOML parser, not a
  regex — a regex-based `version_files` entry for it silently matches nothing) and
  `annotated_tag = true` (commitizen creates lightweight tags by default, and `git push
  --follow-tags` only pushes annotated ones — without this the tag never reaches
  `origin`, even though the bump commit itself does).
- A `concurrency` lock on that workflow means if two merges land close together, the
  second run waits for the first's bump+tag+push to land before computing its own — so
  it's always relative to what's actually on `main`, never a race.
- Net effect: **every merged PR becomes exactly one release**, automatically, with zero
  manual steps. There's no more bundling several PRs into one hand-picked release — if
  you want a bigger jump (e.g. a MAJOR bump), that has to come from the PR's own commit
  type (`feat!:` / a `BREAKING CHANGE:` footer), not from batching.
- `just release` (manual `cz bump` + push) is marked **EMERGENCY-ONLY** in the
  `justfile` — a fallback for when `release.yml` itself is broken or unavailable, not
  part of normal feature work. Running it on a feature branch, or in parallel with
  `release.yml`, can produce duplicate/conflicting version bumps.

## 7. Rules for AI coding assistants (Claude Code or any other agent)

These bind any agent working in these repos, on top of everything above:

1. **Never `git commit` or `git push` to `main`/`master`**, even if the instruction to
   "commit" or "push" doesn't name a branch — treat unqualified instructions as "on my
   current feature branch," never on `main`.
2. **Land changes only via branch → commit → PR.** Never merge a PR yourself unless
   explicitly told to, and never bypass review.
3. **Never push without approval given in that same message/turn.** Local commits on a
   feature branch don't need re-confirmation each time; publishing (`git push`) does,
   every time — prior approval doesn't carry forward to the next push.
4. **Never skip hooks or CI to get a change through**: no `--no-verify`, no editing a
   workflow file to remove a check the change is failing, no `git commit --amend` on a
   commit that's already been pushed/reviewed.
5. **If the current branch is unclear, or whether an action would land on `main` is
   unclear — stop and ask.** Don't guess on anything that touches shared branches.
6. **When you can't or shouldn't act yourself** (no push permission, action needs human
   sign-off), hand back the exact commands: the `git add`, the commit message, and either
   the `git push` + PR command or a `gh pr create` invocation — ready to run, not
   paraphrased.
7. **Prefer new commits over `--amend`/`--force`** once anything is pushed or shared;
   these are fine only on a private, not-yet-pushed local commit.

## 8. Keeping this in sync

This file lives in `terraeye-si-template` and should be copied (or linked) into every
project repo's `docs/`. If you improve a hook, a workflow, or this doc in one place,
port the change to the other — a template and its downstream projects drifting apart is
how "team-wide" standards quietly stop being team-wide.
