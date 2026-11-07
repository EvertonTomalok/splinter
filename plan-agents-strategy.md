# Plan — Estratégia `agents` (plan-with-agents) no Splinter

Porta o fluxo de orquestração do `plan-with-agents` / `agent-engine` (Ama-Hub, Go)
para dentro do Splinter em Python: a estratégia `agents` (alias `splinter`) refina
o PRD em **waves de briefs**, cria uma **branch de integração dedicada**, e executa
os briefs em worktrees paralelas através de um **pool dinâmico de engines** (CLIs
workhorse: mcode, agy, claude, codex, opencode), com **guard shims** que impedem o
workhorse de commitar/rodar builds, **watchdog de inatividade**, **classificação de
falhas** e **fallback** sem queimar tentativas. O Splinter commita, valida
(acceptance check) e faz o merge `--no-ff`.

Fontes: `skills/system/agent-engine` (Ama-Hub) para semântica exata (constantes,
regexes, guard, contrato do workhorse) e o codebase Splinter para convenções
(`strategies/`, `providers/`, `procreg.py`, `vcs/worktree.py`, `agents/`).

## Decisões-chave

1. **Branch de integração dedicada**: `splinter/<slug>-int` em worktree própria
   (`.splinter/worktrees/<slug>-int`); cada brief ramifica dela
   (`splinter/<slug>-<brief-id>`) e entra com merge `--no-ff` (conflito resolvido
   para o lado do brief, validado pelo acceptance). Worktree+branch do brief são
   removidas após o merge.
2. **Guards**: port de `guard.sh` + shims de PATH (`splinter/vcs/guard.py`).
   Bloqueia git write, build, test, linters, geradores, package managers,
   containers, engines aninhados e kill (exit 126, "BLOCKED by the run guard").
   O workhorse apenas edita arquivos; o Splinter commita e valida. Complementos
   nativos: deny lists do claude (`--disallowedTools`), permission JSON do
   opencode (`OPENCODE_CONFIG_CONTENT`), sandbox do codex (`workspace-write`,
   rede off) e `--mode accept-edits --sandbox` do agy.
3. **Acceptance check por brief**: configurável via `agents.acceptance_check`.
   Default `gate+eval` (gate determinístico no worktree + eval leve
   validando diff vs. acceptance). Opções: `gate`, `gate+orchestrator`
   (LLM orquestrador julga diff vs. acceptance).
4. **Pool dinâmico & fallback**: watchdog de inatividade (STALL) no
   `procreg.run_subprocess`; classificação de falha em STALL, LIMIT, CRASH,
   EMPTY, MISSING (`splinter/engines/failures.py`). Falhas de infra ativam o
   circuit breaker do engine e reenviam o mesmo brief ao próximo engine
   disponível na mesma worktree **sem queimar tentativas** do workhorse.
   Orçamento por brief: 3 tentativas (2 retornos de ajuste à mesma sessão);
   esgotado o pool ou as tentativas, o próprio Splinter assume o brief
   (fallback no loop `_run_task_loop` direto na worktree).

## Contratos de módulo (fonte de verdade para implementação)

### `splinter/procreg.py` (edição, M1)

- `run_subprocess(cmd, *, timeout, cwd=None, on_line=None, env=None,
  idle_timeout=None) -> CompletedProcess` — `env` repassa ao `Popen`; 
  `idle_timeout` (segundos) só se aplica com `on_line`: cada linha lida em
  stdout/stderr atualiza `last_activity`; loop principal mata o grupo de
  processos e levanta `IdleTimeoutError(idle_seconds, cmd)` quando
  `now - last_activity > idle_timeout`.
- `class IdleTimeoutError(RuntimeError)` com atributos `idle_seconds: float`,
  `cmd: list[str]`.

### `splinter/engines/failures.py` (novo, M1)

```python
class FailureKind(StrEnum): STALL LIMIT CRASH EMPTY MISSING

@dataclass(frozen=True)
class EngineFailure:
    kind: FailureKind
    detail: str = ""
    exit_code: int | None = None

LIMIT_TAIL_BYTES = 6000
LIMIT_PATTERN = re.compile(
    r"(?i)usage limit|rate.?limit|limit reached|credit balance|quota|"
    r"insufficient|too many requests|\"status\":429| 429 "
)
RESULT_PATTERN = re.compile(r"AGENT_RESULT: ([A-Z]+) ([A-Za-z0-9-]+)")

def find_limit(stdout: str, stderr: str) -> str | None      # regex nos últimos 6000 bytes
def find_result(text: str) -> tuple[str, str] | None         # (status, id) via RESULT_PATTERN
def classify(*, returncode: int | None, stdout: str, stderr: str,
             has_result: bool, stalled: bool = False,
             missing: bool = False) -> EngineFailure | None
```

Precedência em `classify`: MISSING > STALL > LIMIT > CRASH > EMPTY; `None`
quando rc=0, com resultado e sem sinais de infra.

### Providers (M2)

- `splinter/providers/mcode.py` — `mcode exec --cwd <cwd> --model <model>
  --permission smart --timeout <timeout> --output-format stream-json
  [--continue <session>] --file <brief.md> [--file <contract.md>] <prompt>`;
  modelo com prefixo `mcode/` stripped. NDJSON stream-json parseado
  (texto de eventos assistant; usage quando presente). Resultado
  `McodeResult(text, tokens, cost, raw, session_id, cost_indeterminate)`.
  `McodeProvider(ModelProvider)` com `name = "mcode"`.
- `splinter/providers/agy.py` — `agy -p <prompt> --model
  gemini-3.8-flash-<effort> --mode accept-edits --sandbox --add-dir <dir...>
  --output-format stream-json`; `variant` ∈ low|medium|high define o sufixo
  do modelo (default low). `AgyProvider(ModelProvider)` com `name = "agy"`.
- Ambos: `run(..., cwd=None, env=None, idle_timeout=None, files: Sequence[str]
  = (), add_dirs: Sequence[str] = ())` no módulo; passam `env`/`idle_timeout`
  ao `procreg.run_subprocess`. `FileNotFoundError`/exit 127 propagam
  (MISSING em cima). Sem chamadas reais em testes (mock `run_subprocess`).
- `roster.provider_for`: adicionar `mcode/` → `"mcode"`, `agy/` → `"agy"`.
- `claude_cli.run(..., env=None, idle_timeout=None, disallowed_tools=(),
  permission_mode="bypassPermissions", unset_env=())`;
  `opencode.run(..., env=None, idle_timeout=None, standalone=False,
  auto=False, config_content=None)` (config_content → env
  `OPENCODE_CONFIG_CONTENT`); `codex.run(..., env=None, idle_timeout=None,
  sandbox=None, network_access=True)`.

### `splinter/engines/catalog.py` (novo, M3)

```python
@dataclass(frozen=True)
class Engine:
    key: str                  # "mcode" | "mcode-flash" | "agy" | "sonnet" | ...
    label: str
    provider: str             # "mcode" | "agy" | "claude" | "codex" | "opencode"
    model: str                # id completo p/ o provider (prefixo stripped)
    role: str
    default_effort: str = "low"
    supports_continue: bool = False
    supports_model_override: bool = True
    supports_effort: bool = False
    unset_env: tuple[str, ...] = ()   # claude-plan: ANTHROPIC_API_KEY/AUTH_TOKEN
```

Catálogo (11 engines, port fiel): mcode, mcode-flash, agy, sonnet, claude-plan,
codex, glm, opencode-minimax, reviewer-opus, reviewer-fable, reviewer-sonnet
(reviewers: provider claude, default_effort high). `engine_by_key`,
`catalog_table()`, `PROBE_PROMPT = "Reply with the word ok only."`,
`PROBE_UP_PATTERN = re.compile(r"(?i)\bok\b")`.

### `splinter/engines/health.py` (novo, M3)

```python
class BreakerState(StrEnum): CLOSED OPEN HALF_OPEN

@dataclass
class CircuitBreaker:
    failure_threshold: int = 2
    cooldown_s: float = 300.0
    def record_success() / record_failure() / state(now=None) / allow(now=None)

class EngineHealth:                     # registry por engine key
    def allow(key) -> bool / trip(key) / reset(key) / snapshot() -> dict

@dataclass(frozen=True)
class EngineProbe: key: str; up: bool; binary: str; detail: str = ""

def probe(engine: Engine, *, timeout: int = 120) -> EngineProbe
def probe_report(probes) -> str         # tabela markdown (up/down/absent)
```

`probe` resolve o binário no PATH e roda `PROBE_PROMPT` via provider do engine
(chamada real em runtime; mockada em testes). Reviewers são inferidos do sonnet.

### `splinter/engines/pool.py` (novo, M3)

```python
DEFAULT_PREFERENCE = ("mcode", "agy", "sonnet", "codex", "glm",
                      "opencode-minimax", "claude-plan", "mcode-flash")

@dataclass(frozen=True)
class PoolConfig: preference: tuple[str, ...] = DEFAULT_PREFERENCE

class EnginePool:
    def __init__(self, health: EngineHealth, config: PoolConfig = PoolConfig())
    def acquire(self, exclude: set[str] | None = None) -> Engine   # PoolExhausted
    def report_success(key: str) / report_infra_failure(key: str)

class PoolExhausted(RuntimeError)
```

### CLI `splinter engines [--probe] [--only a,b] [--json]` (M3)

Lista o catálogo; com `--probe` faz probe vivo (gasta quota real) e imprime
o relatório; `--json` emite o contrato `{route?, engines: [...]}`.

### `splinter/vcs/guard.py` (novo, M4)

- `GUARDED_NAMES`: os 31 shims do agent-engine (git, go, make, pnpm, npm,
  npx, yarn, vitest, tsc, eslint, next, golangci-lint, uv, pytest, mypy,
  ruff, pip, docker, docker-compose, colima, gh, terraform, claude,
  claude-credit, opencode, agy, mcode, codex, kill, pkill, killall, brew).
- `GUARD_SH`: port verbatim do `guard.sh` (allow-list git status|diff|log|
  show|blame|ls-files|rev-parse|grep|cat-file; go doc|env|version|list|fmt;
  resto deny 126; pass-through remove o dir de shims do PATH).
- `setup_guard(run_dir: Path) -> GuardEnv` — escreve `guard.sh` (0755) e cria
  `shims/<name>` → symlink para guard.sh.
- `guarded_env(guard, *, base=None, unset=(), set_env=None) -> dict[str, str]`
  — PATH com shims primeiro; remove unset; aplica set_env.
- `workhorse_deny() / reviewer_deny() -> list[str]` (deny lists do claude) e
  `opencode_permission_json() -> str` (policy ordenada do opencode).

### `splinter/agents/laws.py` (novo, M4)

- `REPORT_PATH = ".agent/report.md"`;
  `parse_agent_result(text) -> AgentResult(status, ident) | None`.
- `workhorse_contract(laws: str = "", plan_ref: str = "") -> str` — COMMON.md
  portado (contrato workhorse de 7 itens + leis do projeto + referência ao plan).
- `brief_document(brief: "Brief") -> str` — template BRIEF.md
  (Goal / Files you may touch / Change sketch / Acceptance / Report).

### `splinter/vcs/worktree.py` (edição, M4)

- `create_worktree(task_id, base_dir=None, base_ref=None)` — `base_ref`
  aponta o start-point do branch (default HEAD).
- `create_integration_worktree(slug, base_dir=None, base_ref=None) ->
  WorktreeHandle` — branch `splinter/<slug>-int`.
- `merge_no_ff(handle, target_dir, message="") -> bool` — merge `--no-ff`
  do branch do brief na worktree de integração; conflito de conteúdo resolvido
  para o lado do brief (`--theirs`/`git rm --force`), falha não-conteúdo →
  `reset --hard` + `WorktreeMergeConflict`. Retorna False quando não há
  commits a mesclar. Serializado por `_MERGE_SEMAPHORE`.

### `splinter/agents/refiner.py` (novo, M5)

```python
@dataclass
class Brief:
    id: str                  # curto: "q1", "pb3a"
    goal: str = ""
    files: list[str] = field(default_factory=list)
    sketch: str = ""
    acceptance: list[str] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)
    engine: str = ""         # hint opcional de engine key
    task_id: str = ""        # task de origem (US-NNN)

def refine_briefs(tasks, ladder, *, prd_text="", session=None, trace=None,
                  task_index=0) -> list[Brief]     # LLM (template "refine"),
                                                  # fallback 1 brief por task
def brief_waves(briefs: list[Brief]) -> list[list[Brief]]  # níveis topo por deps
def brief_to_task(brief: Brief) -> Task            # adaptador p/ DagScheduler
```

Template `splinter/prompts/refine.md`: recebe tasks+contexto, devolve JSON
`{"briefs": [{id, task, goal, files, sketch, acceptance, deps, engine}]}`;
parse defensivo com validação de deps (ciclo → ValueError).

### `splinter/agents/acceptance.py` (novo, M6)

```python
class AcceptanceMode(StrEnum): GATE GATE_EVAL GATE_ORCHESTRATOR

def parse_acceptance_mode(value: str | None) -> AcceptanceMode
    # None/"" → GATE_EVAL; "gate" | "gate+eval" | "gate+orchestrator"

@dataclass(frozen=True)
class AcceptanceResult:
    passed: bool
    mode: AcceptanceMode
    gate_passed: bool
    detail: str = ""
    verdict: EvalVerdict | None = None

def check_acceptance(brief, *, mode, project_dir, session=None, ladder=None,
                     diff: str = "", trace=None, task_index: int = 0,
                     languages: set[str] | None = None) -> AcceptanceResult
```

- `gate`: `run_gate(project_dir=..., languages=...)`.
- `gate+eval`: gate + eval leve (dispatch `run_text`, template
  `acceptance_eval`, model `ladder.eval_model` @ low) validando diff vs.
  acceptance; VERDICT: PASS/FAIL.
- `gate+orchestrator`: gate + julgamento por LLM orquestrador (role
  `"orchestrator"`), mesmo protocolo VERDICT.

Template `splinter/prompts/acceptance_eval.md`.

### `splinter/strategies/agents.py` (novo, M6)

`@register class AgentsStrategy(Strategy)`, `name = "agents"`,
`aliases = ["splinter"]`. Fluxo do `execute(tasks, session, ladder, **kw)`:

1. `refine_briefs` → briefs; `brief_waves` (log em `loop.md`).
2. Run dir `session.dir/"agents-run"`: `briefs/ logs/ reports/ shims/ .watch/`;
   `setup_guard`; `EngineHealth` + `EnginePool` (preferência de
   `agents.engines`); worktree de integração `splinter/<slug>-int`.
3. `DagScheduler(brief_to_task(b))` + `ThreadPoolExecutor(max_concurrency or
   default_max_concurrency())`; dispatch loop no padrão de
   `cascade._run_parallel_dag` (locks session/git, checkpoint em
   `agents-run/checkpoint.json` só em PASS genuíno).
4. Por brief (`_run_brief`) — máquina de estados
   `PENDING → LAUNCHED → CLASSIFIED → {ACCEPTED, FIX, REROUTE, ASSUMED,
   BLOCKED} → MERGED`: lança engine do pool com prompt workhorse
   (contrato + report `AGENT_RESULT`), env com guard shims, cwd = worktree do
   brief, `idle_timeout = stall_min * 60`. Classifica por exceção/saída
   (`IdleTimeoutError` → STALL; FileNotFoundError/127 → MISSING; rc≠0 →
   CRASH; LIMIT regex; sem AGENT_RESULT → EMPTY). Infra → `report_infra_failure`
   + reroute mesmo brief/worktree ao próximo engine (exclui o falho) sem
   queimar tentativa; `PoolExhausted` → queima tentativa. Sucesso do engine →
   `check_acceptance`; falha → fix brief `<id>-fix{n}.md` retomando a sessão
   (mcode/codex `--continue`), máx 2 retornos. 3ª falha → Splinter assume
   (`_run_task_loop` do DirectStrategy na worktree do brief). PASS →
   `commit_worktree` + `merge_no_ff` na integração + teardown + STATE.
5. Pós-waves: gate final na worktree de integração; RunResults sintéticos por
   brief; eventos em `events.jsonl` via `session.append("events.md", ...)`;
   STATE.md (`agents-run/STATE.md`) com merges e verificação.

### Registro/CLI/config (M6)

- `strategies/registry._ensure_registered()`: importar `agents`.
- `configure.py`: `configured_agents_options() -> AgentsOptions(
  acceptance_check, engines, stall_min, max_attempts, probe_on_start)` lendo
  `agents:` do config; seção documentada no AGENTS.md.
- CLI `engines` (acima). `splinter run --strategy agents` já flui pelo pipeline.

## Marcos

- **M1** procreg watchdog + `engines/failures.py`
- **M2** providers mcode/agy + roster + flags sandbox/env
- **M3** catalog/health/pool + CLI engines
- **M4** guard + laws + worktree (base_ref, integração, merge_no_ff)
- **M5** refiner + templates (refine, acceptance_eval em TEMPLATE_NAMES)
- **M6** acceptance + estratégia agents + registro/CLI/config
- **M7** AGENTS.md + `samples/agents-prd.md`
- **M8** configure como dono do pool: `write_agents_config` + flags CLI
  (`--agents-engines/--agents-acceptance/--agents-engine-model/--agents-engine-effort/…`),
  seção "Agents engine pool" no TUI, `engine_models`/`engine_efforts` aplicados
  pelo `EnginePool` (`resolve_engine`), e **pool por nível de estratégia**:
  `models.tiers[n]` aceita lista ordenada, `resolve_model` pula modelos com
  breaker aberto (`engines/runtime.py`), e o `run_task` reroteia para o próximo
  modelo sadio do mesmo tier em gaps transientes (429/overload) — estratégias
  existentes (direct/cascade/adaptive/sprint) mais inteligentes nos modelos
  chamados, sem mudar seu fluxo.

## Regras do repo (gate final)

- `uv run ruff check && uv run mypy splinter && uv run pytest` — zero erros.
- Proibido teste que chame modelo real ou rode subprocesso de CLI externa:
  mockar `procreg.run_subprocess`, providers, `dispatch.run_text`,
  `_make_plan`, worktree/git. Testes em segundos.
- Python 3.11+, `from __future__ import annotations`, hints em tudo
  (`mypy --strict`), dataclasses, sem comentários, linha ≤ 100, ruff E/F/W/I.
