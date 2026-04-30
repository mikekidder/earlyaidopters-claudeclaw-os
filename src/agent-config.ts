import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

import { CLAUDECLAW_CONFIG, PROJECT_ROOT, STORE_DIR } from './config.js';
import { readEnvFile } from './env.js';

export const DEFAULT_MAIN_DESCRIPTION = 'Primary ClaudeClaw bot';

function mainConfigPath(): string {
  return path.join(STORE_DIR, 'main-config.json');
}

export interface AgentConfig {
  name: string;
  description: string;
  botTokenEnv: string;
  botToken: string;
  model?: string;
  mcpServers?: string[];
  obsidian?: {
    vault: string;
    folders: string[];
    readOnly?: string[];
  };
  /** Pika voice id used when this agent joins a video meeting. Falls back
   *  to the Pika preset English_radiant_girl if unset. */
  meetVoiceId?: string;
  /** Display name shown in the meeting ("Your Agent wants to join"). Falls
   *  back to the agent's name or id with first letter capitalized. */
  meetBotName?: string;
  /** Restrict which user-invocable skills this bot exposes as slash
   *  commands. When set, only these skill names appear in the Telegram
   *  menu and are dispatchable via /<name>. When absent, the bot sees
   *  every user_invocable skill under ~/.claude/skills/. */
  skillsAllowlist?: string[];
}

/**
 * Resolve the directory for a given agent, checking CLAUDECLAW_CONFIG first,
 * then falling back to PROJECT_ROOT/agents/<id>.
 */
export function resolveAgentDir(agentId: string): string {
  const externalDir = path.join(CLAUDECLAW_CONFIG, 'agents', agentId);
  if (fs.existsSync(path.join(externalDir, 'agent.yaml'))) {
    return externalDir;
  }
  return path.join(PROJECT_ROOT, 'agents', agentId);
}

/**
 * Resolve the CLAUDE.md path for a given agent, checking CLAUDECLAW_CONFIG first,
 * then falling back to PROJECT_ROOT/agents/<id>/CLAUDE.md.
 */
export function resolveAgentClaudeMd(agentId: string): string | null {
  const externalPath = path.join(CLAUDECLAW_CONFIG, 'agents', agentId, 'CLAUDE.md');
  if (fs.existsSync(externalPath)) {
    return externalPath;
  }
  const repoPath = path.join(PROJECT_ROOT, 'agents', agentId, 'CLAUDE.md');
  if (fs.existsSync(repoPath)) {
    return repoPath;
  }
  return null;
}

export function loadAgentConfig(agentId: string): AgentConfig {
  const agentDir = resolveAgentDir(agentId);
  const configPath = path.join(agentDir, 'agent.yaml');

  if (!fs.existsSync(configPath)) {
    throw new Error(`Agent config not found: ${configPath}`);
  }

  const raw = yaml.load(fs.readFileSync(configPath, 'utf-8')) as Record<string, unknown>;

  const name = raw['name'] as string;
  const description = (raw['description'] as string) ?? '';
  const botTokenEnv = raw['telegram_bot_token_env'] as string;
  const model = raw['model'] as string | undefined;

  if (!name) {
    throw new Error(`Agent config ${configPath} must have 'name'`);
  }

  // Main agent uses TELEGRAM_BOT_TOKEN directly; specialist agents declare
  // their own env var via telegram_bot_token_env.
  const effectiveTokenEnv = botTokenEnv || (agentId === 'main' ? 'TELEGRAM_BOT_TOKEN' : '');
  if (!effectiveTokenEnv) {
    throw new Error(`Agent config ${configPath} must have 'telegram_bot_token_env'`);
  }

  const env = readEnvFile([effectiveTokenEnv]);
  const botToken = process.env[effectiveTokenEnv] || env[effectiveTokenEnv] || '';
  if (!botToken) {
    throw new Error(`Bot token not found: set ${effectiveTokenEnv} in .env`);
  }

  let obsidian: AgentConfig['obsidian'];
  const obsRaw = raw['obsidian'] as Record<string, unknown> | undefined;
  if (obsRaw) {
    const vault = obsRaw['vault'] as string;
    if (vault && !fs.existsSync(vault)) {
      // eslint-disable-next-line no-console
      console.warn(`[${agentId}] WARNING: Obsidian vault path does not exist: ${vault}`);
      console.warn(`[${agentId}] Update obsidian.vault in agent.yaml to your local vault path.`);
    }
    obsidian = {
      vault,
      folders: (obsRaw['folders'] as string[]) ?? [],
      readOnly: (obsRaw['read_only'] as string[]) ?? [],
    };
  }

  const mcpServers = raw['mcp_servers'] as string[] | undefined;
  const meetVoiceId = typeof raw['meet_voice_id'] === 'string' ? (raw['meet_voice_id'] as string) : undefined;
  const meetBotName = typeof raw['meet_bot_name'] === 'string' ? (raw['meet_bot_name'] as string) : undefined;
  const rawSkills = raw['skills_allowlist'];
  const skillsAllowlist = Array.isArray(rawSkills)
    ? (rawSkills as unknown[]).filter((s): s is string => typeof s === 'string').map((s) => s.toLowerCase())
    : undefined;

  return {
    name,
    description,
    botTokenEnv: effectiveTokenEnv,
    botToken,
    model,
    mcpServers,
    obsidian,
    meetVoiceId,
    meetBotName,
    skillsAllowlist,
  };
}

/** Update the model field in an agent's agent.yaml file. */
export function setAgentModel(agentId: string, model: string): void {
  const agentDir = resolveAgentDir(agentId);
  const configPath = path.join(agentDir, 'agent.yaml');
  if (!fs.existsSync(configPath)) throw new Error(`Agent config not found: ${configPath}`);

  const raw = yaml.load(fs.readFileSync(configPath, 'utf-8')) as Record<string, unknown>;
  raw['model'] = model;
  fs.writeFileSync(configPath, yaml.dump(raw, { lineWidth: -1 }), 'utf-8');
}

/** Update the description field in an agent's agent.yaml file. */
export function setAgentDescription(agentId: string, description: string): void {
  const trimmed = description.trim();
  if (!trimmed) throw new Error('description cannot be empty');

  const agentDir = resolveAgentDir(agentId);
  const configPath = path.join(agentDir, 'agent.yaml');
  if (!fs.existsSync(configPath)) throw new Error(`Agent config not found: ${configPath}`);

  const raw = yaml.load(fs.readFileSync(configPath, 'utf-8')) as Record<string, unknown>;
  raw['description'] = trimmed;
  fs.writeFileSync(configPath, yaml.dump(raw, { lineWidth: -1 }), 'utf-8');
}

/** Load the description for the main bot (persisted, editable). */
export function getMainDescription(): string {
  const configPath = mainConfigPath();
  try {
    if (!fs.existsSync(configPath)) return DEFAULT_MAIN_DESCRIPTION;
    const raw = JSON.parse(fs.readFileSync(configPath, 'utf-8')) as { description?: string };
    const desc = (raw.description ?? '').trim();
    return desc || DEFAULT_MAIN_DESCRIPTION;
  } catch {
    return DEFAULT_MAIN_DESCRIPTION;
  }
}

/** Persist a description for the main bot. */
export function setMainDescription(description: string): void {
  const trimmed = description.trim();
  if (!trimmed) throw new Error('description cannot be empty');

  if (!fs.existsSync(STORE_DIR)) fs.mkdirSync(STORE_DIR, { recursive: true });

  const configPath = mainConfigPath();
  let raw: Record<string, unknown> = {};
  if (fs.existsSync(configPath)) {
    try { raw = JSON.parse(fs.readFileSync(configPath, 'utf-8')) as Record<string, unknown>; } catch { raw = {}; }
  }
  raw['description'] = trimmed;
  fs.writeFileSync(configPath, JSON.stringify(raw, null, 2) + '\n', 'utf-8');
}

/** List all configured agent IDs (directories under agents/ with agent.yaml).
 *  Scans both CLAUDECLAW_CONFIG/agents/ and PROJECT_ROOT/agents/, deduplicating.
 */
export function listAgentIds(): string[] {
  const ids = new Set<string>();

  for (const baseDir of [
    path.join(CLAUDECLAW_CONFIG, 'agents'),
    path.join(PROJECT_ROOT, 'agents'),
  ]) {
    if (!fs.existsSync(baseDir)) continue;
    for (const d of fs.readdirSync(baseDir)) {
      if (d.startsWith('_')) continue;
      const yamlPath = path.join(baseDir, d, 'agent.yaml');
      if (fs.existsSync(yamlPath)) ids.add(d);
    }
  }

  return [...ids];
}

// ── War Room roster ──────────────────────────────────────────────────────

export interface WarRoomAgent {
  id: string;
  name: string;
  description: string;
  role: string;
  /** Condensed persona extracted from the agent's CLAUDE.md (personality + role sections). */
  persona?: string;
}

/** Default 5-agent roster shown when no custom agent.yaml files are configured. */
export const DEFAULT_WARROOM_ROSTER: WarRoomAgent[] = [
  { id: 'main', name: 'Main', description: 'General ops and triage', role: 'The Hand of the King' },
  { id: 'research', name: 'Research', description: 'Deep web research, academic sources, competitive intel, trend analysis', role: 'Grand Maester' },
  { id: 'comms', name: 'Comms', description: 'Email, Slack, Telegram, WhatsApp, customer comms, inbox triage', role: 'Master of Whisperers' },
  { id: 'content', name: 'Content', description: 'Writing, YouTube scripts, LinkedIn posts, blog copy, creative direction', role: 'The Royal Bard' },
  { id: 'ops', name: 'Ops', description: 'Calendar, scheduling, cron, system operations, MCP tool work, automations', role: 'Master of War' },
];

/** Sections in CLAUDE.md that are operational or not needed for War Room persona. Skip these. */
const SKIP_SECTIONS = new Set([
  'hive mind', 'scheduling tasks', 'scheduling', 'rules',
  'obsidian access', 'obsidian', 'your obsidian folders', 'your environment',
  'available skills', 'message format', 'memory', 'special commands',
  'mission tasks', 'sending files via telegram', 'launchd rules',
  // "Who Is X" sections provide user context that's useful for Claude Code
  // but not needed in the War Room voice persona (eats char budget).
  'who is mike',
]);

/**
 * Extract a condensed persona from an agent's CLAUDE.md.
 * Keeps personality, role, scope, and identity sections.
 * Strips operational sections (hive mind, scheduling, rules, etc.)
 * and code blocks. Returns null if no CLAUDE.md found.
 */
function extractPersonaFromClaudeMd(agentId: string): string | null {
  const mdPath = resolveAgentClaudeMd(agentId);
  if (!mdPath) return null;

  try {
    const raw = fs.readFileSync(mdPath, 'utf-8');
    const lines = raw.split('\n');
    const kept: string[] = [];
    let skipping = false;
    let inCodeBlock = false;

    for (const line of lines) {
      // Track code fences
      if (line.trim().startsWith('```')) {
        inCodeBlock = !inCodeBlock;
        continue;
      }
      if (inCodeBlock) continue;

      // Check for section headers
      const headerMatch = line.match(/^#{1,3}\s+(.+)/);
      if (headerMatch) {
        const sectionName = headerMatch[1].trim().toLowerCase();
        // "Who Is X" sections (any name) provide user context not needed in War Room
        skipping = SKIP_SECTIONS.has(sectionName) || sectionName.startsWith('who is');
        if (skipping) continue;
        // Keep the header text but strip the markdown prefix
        kept.push(headerMatch[1].trim());
        continue;
      }

      if (skipping) continue;

      // Keep non-empty content lines
      const trimmed = line.trim();
      if (trimmed) {
        // Strip markdown bold/italic but keep the text
        kept.push(trimmed.replace(/\*\*(.+?)\*\*/g, '$1').replace(/\*(.+?)\*/g, '$1'));
      }
    }

    const persona = kept.join('\n').trim();
    // Cap at ~2000 chars to keep Gemini Live responsive while ensuring
    // routing-critical sections (Scope, Your Role) make it in.
    return persona.length > 2000 ? persona.slice(0, 2000) + '...' : persona;
  } catch {
    return null;
  }
}

/**
 * Build the War Room agent roster using a two-tier fallback:
 *  - CONFIGURED: if listAgentIds() finds agent.yaml files, use those agents
 *  - DEFAULT: otherwise return the 5-agent demo roster
 */
export function buildWarRoomRoster(): WarRoomAgent[] {
  const ids = listAgentIds();
  if (ids.length === 0) {
    return DEFAULT_WARROOM_ROSTER;
  }

  // Ensure main-equivalent is first
  const ordered = ids.includes('main')
    ? ['main', ...ids.filter((id) => id !== 'main')]
    : ids;

  return ordered.map((id) => {
    const caps = getAgentCapabilities(id);
    const defaultEntry = DEFAULT_WARROOM_ROSTER.find((d) => d.id === id);
    const persona = extractPersonaFromClaudeMd(id) || undefined;
    const agentName = caps?.name || id.charAt(0).toUpperCase() + id.slice(1);
    // Only use the default GoT role when the agent name hasn't been customised.
    // If "main" was renamed to "ObiPrime", "The Hand of the King" doesn't apply.
    const isCustomName = defaultEntry && agentName !== defaultEntry.name;
    return {
      id,
      name: agentName,
      description: caps?.description || defaultEntry?.description || '',
      role: isCustomName
        ? (caps?.description || 'Specialist')
        : (defaultEntry?.role || caps?.description || 'Specialist'),
      persona,
    };
  });
}

/** Return the capabilities (name + description) for a specific agent. */
export function getAgentCapabilities(
  agentId: string,
): { name: string; description: string } | null {
  try {
    const config = loadAgentConfig(agentId);
    return { name: config.name, description: config.description };
  } catch {
    return null;
  }
}

/**
 * List all configured agents with their descriptions.
 * Unlike `listAgentIds()`, this returns richer metadata and silently
 * skips agents whose config fails to load (e.g. missing token).
 */
export function listAllAgents(): Array<{
  id: string;
  name: string;
  description: string;
  model?: string;
}> {
  const ids = listAgentIds();
  const result: Array<{
    id: string;
    name: string;
    description: string;
    model?: string;
  }> = [];

  for (const id of ids) {
    try {
      const config = loadAgentConfig(id);
      result.push({
        id,
        name: config.name,
        description: config.description,
        model: config.model,
      });
    } catch {
      // Skip agents with broken config
    }
  }

  return result;
}
