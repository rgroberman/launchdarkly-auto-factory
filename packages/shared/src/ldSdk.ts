/**
 * Native LaunchDarkly SDK bootstrap for the Phase 1 runtime.
 *
 * This is the LaunchDarkly-native foundation the prototype runs on:
 *   - the Node *server SDK* (`@launchdarkly/node-server-sdk`) evaluates flags
 *     (e.g. the AI-provider selector), and
 *   - the Node *AI SDK* (`@launchdarkly/server-sdk-ai`) resolves agent configs
 *     and agent graphs (interpolated instructions + model + generation tracking).
 *
 * Both share one server-SDK client, created from the `LD_SDK_KEY` (the `sdk-`
 * key — distinct from the `api-` PAT used for REST writes like flag creation).
 * The client is environment-scoped to the FACTORY/control-plane project that
 * holds the AI configs, graph, and operational flags.
 */

import { type LDClient, type LDContext, type LDOptions, init } from "@launchdarkly/node-server-sdk";
import { type LDAIClient, initAi } from "@launchdarkly/server-sdk-ai";
import { loadDotEnv } from "./env.js";

/**
 * Build server-SDK options. By default the SDK targets the commercial cloud
 * (stream/sdk/events.launchdarkly.com). For instances such as catamorphic, set
 * LD_STREAM_URL / LD_SDK_BASE_URL / LD_EVENTS_URL to override the relevant
 * service endpoints. Any subset may be provided; unset endpoints keep defaults.
 *
 * The node-server-sdk (v9) exposes these as the flat baseUri/streamUri/eventsUri
 * options rather than serviceEndpoints.
 */
function buildSdkOptions(): LDOptions {
  const streamUri = process.env.LD_STREAM_URL;
  const baseUri = process.env.LD_SDK_BASE_URL;
  const eventsUri = process.env.LD_EVENTS_URL;
  return {
    ...(baseUri ? { baseUri } : {}),
    ...(streamUri ? { streamUri } : {}),
    ...(eventsUri ? { eventsUri } : {}),
  };
}

export interface LdSdk {
  /** Server SDK client — flag evaluation. */
  ldClient: LDClient;
  /** AI SDK client — agent configs + agent graphs + tracking. */
  aiClient: LDAIClient;
}

let cached: LdSdk | null = null;

/** Initialize (once) and return the shared LaunchDarkly server + AI SDK clients. */
export async function getLdSdk(): Promise<LdSdk> {
  if (cached) return cached;
  loadDotEnv();
  const sdkKey = process.env.LD_SDK_KEY;
  if (!sdkKey) {
    throw new Error("LD_SDK_KEY not set — the server SDK key for flag evaluation and AI config/graph resolution");
  }
  const ldClient = init(sdkKey, buildSdkOptions());
  await ldClient.waitForInitialization({ timeout: 15 });
  const aiClient = initAi(ldClient);
  cached = { ldClient, aiClient };
  return cached;
}

/** Flush and close the SDK so the process can exit (the client holds the event loop open). */
export async function closeLdSdk(): Promise<void> {
  if (!cached) return;
  try {
    await cached.ldClient.flush();
  } catch {
    /* best-effort */
  }
  await cached.ldClient.close();
  cached = null;
}

/**
 * The LaunchDarkly context for this pipeline run. Used both for flag evaluation
 * and as the targeting context for AI config/graph resolution, so a customer can
 * target rules at specific repos/pipelines later.
 */
export function pipelineContext(extra: Record<string, unknown> = {}): LDContext {
  return {
    kind: "service",
    key: process.env.LD_PIPELINE_CONTEXT_KEY ?? "auto-factory-phase1",
    name: "AutoFactory Phase 1",
    ...extra,
  } as LDContext;
}
