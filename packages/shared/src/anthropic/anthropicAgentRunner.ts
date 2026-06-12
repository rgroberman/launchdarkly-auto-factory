/**
 * Anthropic implementation of the `AgentRunner` seam.
 *
 * Runs the AutoFactory agent graph LOCALLY. The agent's instructions and model
 * are resolved NATIVELY by the LaunchDarkly AI SDK (the graph walker passes the
 * already-interpolated `instructions`, `model`, and a per-node `tracker` from the
 * `LDAIAgentConfig`). This runner just executes: it drives an Anthropic tool-use
 * loop with a read-only sandbox tool set (see sandboxTools.ts) and records
 * generation metrics to LaunchDarkly via the tracker.
 *
 * This is the piece Vega's `agentDispatch` does NOT do — Vega runs built-in
 * personas and ignores the AI config's instructions. Here the LD-authored
 * instructions ARE the agent.
 *
 * Sandbox / dry-run: the agents cannot write to LaunchDarkly, git, or the repo.
 * They inspect the code and emit routing tags via `tag_conversation`, which is
 * what the graph walker needs to advance the chain. Promote to write tools (LD
 * flag creation, git) for in-pipeline runs without touching the walker.
 */

import Anthropic from "@anthropic-ai/sdk";
import type { AgentNodeRequest, AgentNodeResult, AgentRunner, AgentStatus } from "../agentRunner.js";
import type { LdResourceWriter } from "./ldWriter.js";
import { SandboxToolExecutor, type ToolCapabilities, buildSandboxTools } from "./sandboxTools.js";

const TAGGING_NOTE = `

You MUST call \`tag_conversation\` with the routing tag(s) your instructions specify
(e.g. flag_created, skip_flagging, flag_worthy, needs_tests, review_approved,
risk_level). The downstream chain advances on these tags — a step that sets no tags
stalls the pipeline.`;

/** Build the execution-mode note appended to the agent's instructions, per capabilities. */
function modeNote(caps: ToolCapabilities): string {
  const lines = [
    "\n\n---\n## EXECUTION MODE",
    "You have read-only repo tools (`read_file`, `list_dir`, `grep`).",
  ];
  if (caps.createFlag) {
    lines.push(
      "You have `create_flag` — creates a REAL boolean flag in the LaunchDarkly app project (idempotent; safe on PR re-runs). When your rules say a flag is needed, CALL it.",
    );
  }
  if (caps.createMetric) {
    lines.push(
      "You have `create_metric` — creates a REAL guarded-release metric in the LaunchDarkly app project (idempotent). To make a metric measurable you must FIRST instrument its event in code: add a LaunchDarkly `track(event_key, …)` call on the path the flag wraps (via `edit_file`), then call `create_metric` with the matching `event_key`. Creating the metric before signals flow is expected.",
    );
  }
  if (caps.editFiles) {
    lines.push(
      "You have `write_file`, `edit_file`, `run_tests`, and `commit_and_push`. EXECUTE your job for real: make the file changes your instructions describe (e.g. wire the flag into the code, or add the test file). If you wrote or changed tests, call `run_tests` to confirm they pass and FIX any failures before committing. Then call `commit_and_push` ONCE to land your changes on the PR branch. Match the existing code patterns you find.",
    );
  } else {
    lines.push("You CANNOT edit files or push commits — describe what you would change and tag accordingly.");
  }
  lines.push("Keep exploration focused, then finish with a short brief for the next agent.");
  return lines.join("\n") + TAGGING_NOTE;
}

const DEFAULT_MAX_TURNS = 30;
const MAX_TOKENS = 4096;
const DEFAULT_MODEL = "claude-sonnet-4-6";

/**
 * FALLBACK per-node capability grants, used only when the graph edge doesn't
 * declare a `capabilities` array (see `resolveGrant`). Prefer putting grants on
 * the graph edges so "which agent can write" is config, not code — this map is a
 * safety net for graphs that predate that and is keyed by config key (so it
 * silently misses renamed agents; the per-node log makes that diagnosable).
 */
const NODE_CAPABILITIES: Record<string, ToolCapabilities> = {
  "autofactory-flag-implementer": { createFlag: true, createMetric: false, editFiles: true },
  "autofactory-flag-testing": { createFlag: false, createMetric: false, editFiles: true },
  // The metrics author creates LD metrics and instruments the event (track()) that
  // feeds them — so it needs create_metric AND edit_files.
  "autofactory-metrics-author": { createFlag: false, createMetric: true, editFiles: true },
};

/** Capability tokens recognized on a graph edge's `capabilities` array. */
export const CAP_CREATE_FLAG = "create_flag";
export const CAP_CREATE_METRIC = "create_metric";
export const CAP_EDIT_FILES = "edit_files";

/**
 * Resolve a node's requested capability grant: from the edge `capabilities` list
 * when present, else the `NODE_CAPABILITIES` fallback, else read-only. Returns the
 * grant + its source for logging (NOT yet intersected with what's globally enabled).
 */
export function resolveGrant(
  configKey: string,
  capabilities: string[] | undefined,
): { grant: ToolCapabilities; source: "edge" | "fallback" | "none" } {
  if (capabilities) {
    return {
      grant: {
        createFlag: capabilities.includes(CAP_CREATE_FLAG),
        createMetric: capabilities.includes(CAP_CREATE_METRIC),
        editFiles: capabilities.includes(CAP_EDIT_FILES),
      },
      source: "edge",
    };
  }
  const fallback = NODE_CAPABILITIES[configKey];
  if (fallback) return { grant: fallback, source: "fallback" };
  return { grant: { createFlag: false, createMetric: false, editFiles: false }, source: "none" };
}

export interface AnthropicAgentRunnerOptions {
  /** Absolute path the sandbox tools operate within (the repo under review / the checkout). */
  sandboxRoot: string;
  /** Anthropic API key; falls back to ANTHROPIC_API_KEY in the env. */
  apiKey?: string;
  /** When provided, `create_flag` is enabled for capable nodes (real flags in the app project). */
  writer?: LdResourceWriter;
  /** When true, file-edit + commit/push tools are enabled for capable nodes. */
  codeChangesEnabled?: boolean;
  /** PR head branch the git tools push to (passed to the sandbox executor). */
  prBranch?: string;
  /** PR base ref the git_diff tool diffs against (passed to the sandbox executor). */
  prBaseRef?: string;
}

export class AnthropicAgentRunner implements AgentRunner {
  private readonly client: Anthropic;

  constructor(private readonly opts: AnthropicAgentRunnerOptions) {
    this.client = new Anthropic(opts.apiKey ? { apiKey: opts.apiKey } : {});
  }

  async runNode(req: AgentNodeRequest): Promise<AgentNodeResult> {
    // Effective capabilities = this node's grant ∩ globally-enabled features.
    const { grant, source } = resolveGrant(req.configKey, req.capabilities);
    const caps: ToolCapabilities = {
      createFlag: grant.createFlag && this.opts.writer !== undefined,
      createMetric: grant.createMetric && this.opts.writer !== undefined,
      editFiles: grant.editFiles && this.opts.codeChangesEnabled === true,
    };
    // Per-node diagnostic: makes a renamed/added agent that silently lost its
    // grant (source "none", read-only) visible in the run logs.
    console.log(
      `[node] ${req.configKey} grant(${source}): createFlag=${grant.createFlag} createMetric=${grant.createMetric} editFiles=${grant.editFiles} → effective createFlag=${caps.createFlag} createMetric=${caps.createMetric} editFiles=${caps.editFiles}`,
    );
    const writer = caps.createFlag || caps.createMetric ? this.opts.writer : undefined;

    const system = (req.instructions ?? "") + modeNote(caps);
    const model = anthropicModelId(req.model);
    const executor = new SandboxToolExecutor(
      this.opts.sandboxRoot,
      writer,
      caps.editFiles,
      this.opts.prBranch,
      this.opts.prBaseRef,
    );
    const tools = buildSandboxTools(caps) as Anthropic.Tool[];
    const maxTurns = req.maxTurns ?? DEFAULT_MAX_TURNS;

    const messages: Anthropic.MessageParam[] = [{ role: "user", content: req.prompt }];
    let finalText = "";
    let status: AgentStatus = "completed";
    let inputTokens = 0;
    let outputTokens = 0;
    const started = Date.now();

    try {
      for (let turn = 0; turn < maxTurns; turn++) {
        const resp = await this.client.messages.create({
          model,
          max_tokens: MAX_TOKENS,
          system,
          tools,
          messages,
        });
        inputTokens += resp.usage.input_tokens;
        outputTokens += resp.usage.output_tokens;
        messages.push({ role: "assistant", content: resp.content });
        finalText = textOf(resp.content) || finalText;

        if (resp.stop_reason !== "tool_use") break;

        const toolUses = resp.content.filter((b): b is Anthropic.ToolUseBlock => b.type === "tool_use");
        const results: Anthropic.ToolResultBlockParam[] = [];
        for (const b of toolUses) {
          const r = await executor.execute(b.name, (b.input ?? {}) as Record<string, unknown>);
          results.push({
            type: "tool_result",
            tool_use_id: b.id,
            content: r.content,
            ...(r.isError ? { is_error: true } : {}),
          });
        }
        messages.push({ role: "user", content: results });

        if (turn === maxTurns - 1) status = "stopped"; // hit the turn cap mid-task
      }
      req.tracker?.trackSuccess();
    } catch (e) {
      status = "failed";
      finalText = e instanceof Error ? e.message : String(e);
      req.tracker?.trackError();
    } finally {
      req.tracker?.trackDuration(Date.now() - started);
      if (inputTokens || outputTokens) {
        req.tracker?.trackTokens({ input: inputTokens, output: outputTokens, total: inputTokens + outputTokens });
      }
    }

    return {
      status,
      messages: [{ role: "assistant", content: finalText, isFinal: true }],
      tags: { ...executor.tags },
    };
  }
}

/** Concatenate the text blocks of an Anthropic response. */
function textOf(content: Anthropic.ContentBlock[]): string {
  return content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("\n")
    .trim();
}

/**
 * Map a LaunchDarkly model name to an Anthropic model id. LD model names may be
 * provider-qualified (e.g. "Anthropic.claude-sonnet-4-6") or Bedrock-style
 * region-qualified (e.g. "us.anthropic.claude-sonnet-4-6-v1:0"). Strip at most an
 * optional leading region segment and a single "anthropic." prefix; everything
 * else (including multi-dot model ids like "...-v1:0") passes through unchanged.
 */
export function anthropicModelId(name: string | undefined): string {
  if (!name) return DEFAULT_MODEL;
  const id = name
    .trim()
    .replace(/^[a-z]{2}\./i, "") // optional region segment, e.g. "us."
    .replace(/^anthropic\./i, ""); // single provider prefix
  return id.trim() || DEFAULT_MODEL;
}
