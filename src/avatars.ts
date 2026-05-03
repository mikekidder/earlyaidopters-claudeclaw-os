import fs from 'fs';
import path from 'path';

import { PROJECT_ROOT, STORE_DIR } from './config.js';
import { resolveAgentDir, loadAgentConfig } from './agent-config.js';
import { logger } from './logger.js';

// Single source of truth for agent avatars. All HTTP endpoints, the
// browser dashboard, and the Daily.co Python spawner go through this
// resolver. The pre-existing two-backend split (curated war-room art
// vs Mission Control's Telegram-aware fallback) caused user-uploaded
// avatars to vanish from War Room views, since War Room only served
// repo-bundled PNGs.

// Source priority:
//   1. Mutable user-owned avatar (uploaded or Telegram-cached)
//        - main:    STORE_DIR/avatars/main.png
//        - others:  resolveAgentDir(id)/avatar.png
//   2. Bundled meet-optimized variant (only for ctx.context === 'meet')
//        warroom/avatars/<id>-meet.png
//   3. Bundled default art
//        warroom/avatars/<id>.png
//   4. null → caller renders initials / 204

export interface AvatarResolveCtx {
  context?: 'default' | 'meet';
}

export interface ResolvedAvatar {
  absPath: string;
  mtimeMs: number;
  size: number;
  source: 'user' | 'bundled-meet' | 'bundled';
}

const ID_RE = /^[a-z0-9_-]+$/i;
const NO_AVATAR_TTL_MS = 24 * 60 * 60 * 1000;

function bundledPath(agentId: string, variant: 'default' | 'meet'): string {
  const suffix = variant === 'meet' ? `${agentId}-meet.png` : `${agentId}.png`;
  return path.join(PROJECT_ROOT, 'warroom', 'avatars', suffix);
}

/** The mutable, runtime-writable avatar location for an agent. Never
 *  writes to warroom/avatars/ — that namespace is bundled, immutable
 *  art tracked in git. */
export function getMutableAvatarPath(agentId: string): string {
  if (!ID_RE.test(agentId)) throw new Error(`invalid agent id: ${agentId}`);
  if (agentId === 'main') {
    return path.join(STORE_DIR, 'avatars', 'main.png');
  }
  return path.join(resolveAgentDir(agentId), 'avatar.png');
}

function noAvatarFlagPath(agentId: string): string {
  if (agentId === 'main') {
    return path.join(STORE_DIR, 'avatars', '.main.no-avatar');
  }
  return path.join(resolveAgentDir(agentId), '.no-avatar');
}

function statSilent(p: string): fs.Stats | null {
  try { return fs.statSync(p); } catch { return null; }
}

/** Resolve the file an HTTP/Daily client should serve for this agent.
 *  Returns null if even the bundled fallback is missing — the caller
 *  decides whether to 204 (browser will render initials) or skip. */
export function resolveAgentAvatar(
  agentId: string,
  ctx: AvatarResolveCtx = {},
): ResolvedAvatar | null {
  if (!ID_RE.test(agentId)) return null;

  const mutable = getMutableAvatarPath(agentId);
  const mutableStat = statSilent(mutable);
  if (mutableStat && mutableStat.isFile()) {
    return {
      absPath: mutable,
      mtimeMs: mutableStat.mtimeMs,
      size: mutableStat.size,
      source: 'user',
    };
  }

  if (ctx.context === 'meet') {
    const meet = bundledPath(agentId, 'meet');
    const meetStat = statSilent(meet);
    if (meetStat && meetStat.isFile()) {
      return {
        absPath: meet,
        mtimeMs: meetStat.mtimeMs,
        size: meetStat.size,
        source: 'bundled-meet',
      };
    }
  }

  const def = bundledPath(agentId, 'default');
  const defStat = statSilent(def);
  if (defStat && defStat.isFile()) {
    return {
      absPath: def,
      mtimeMs: defStat.mtimeMs,
      size: defStat.size,
      source: 'bundled',
    };
  }

  return null;
}

/** Quoted weak ETag derived from mtime + size. Cheap, stable across
 *  process restarts, and changes the moment a write lands. */
export function avatarEtag(r: ResolvedAvatar): string {
  return `W/"${Math.floor(r.mtimeMs)}-${r.size}"`;
}

/** Best-effort etag for the agents-list response (cache-bust query
 *  param). Falls back to '0' so the URL still parses when nothing
 *  resolves. */
export function avatarEtagForId(agentId: string, ctx: AvatarResolveCtx = {}): string {
  const r = resolveAgentAvatar(agentId, ctx);
  if (!r) return '0';
  return `${Math.floor(r.mtimeMs)}-${r.size}`;
}

// ── Per-agent write mutex ─────────────────────────────────────────────
const writeChains = new Map<string, Promise<unknown>>();

function withAgentLock<T>(agentId: string, fn: () => Promise<T>): Promise<T> {
  const prev = writeChains.get(agentId) ?? Promise.resolve();
  const next = prev.then(fn, fn);
  writeChains.set(agentId, next.catch(() => {}));
  return next;
}

// ── Telegram on-demand fetch ──────────────────────────────────────────
// Existing code lived inline in dashboard.ts:2634-2667. Pulled out
// verbatim so HTTP and CLI paths share it. Writes only to the mutable
// path; never touches warroom/avatars/.
export async function tryFetchTelegramAvatar(agentId: string): Promise<boolean> {
  if (!ID_RE.test(agentId) || agentId === 'main') return false;
  let botToken: string;
  try {
    botToken = loadAgentConfig(agentId).botToken;
  } catch {
    return false;
  }
  if (!botToken) return false;

  return withAgentLock(agentId, async () => {
    // Re-check after acquiring lock — earlier caller may have populated.
    const mutable = getMutableAvatarPath(agentId);
    if (fs.existsSync(mutable)) return true;

    const flag = noAvatarFlagPath(agentId);
    if (fs.existsSync(flag)) {
      const age = Date.now() - fs.statSync(flag).mtimeMs;
      if (age < NO_AVATAR_TTL_MS) return false;
      try { fs.unlinkSync(flag); } catch {}
    }

    try {
      const meRes = await fetch(`https://api.telegram.org/bot${botToken}/getMe`);
      const meJson: any = await meRes.json();
      const smallId = meJson?.result?.photo?.small_file_id;
      if (!smallId) {
        try {
          fs.mkdirSync(path.dirname(flag), { recursive: true });
          fs.writeFileSync(flag, '');
        } catch {}
        return false;
      }
      const fileRes = await fetch(
        `https://api.telegram.org/bot${botToken}/getFile?file_id=${encodeURIComponent(smallId)}`,
      );
      const fileJson: any = await fileRes.json();
      const filePath = fileJson?.result?.file_path;
      if (!filePath) return false;

      const dlRes = await fetch(`https://api.telegram.org/file/bot${botToken}/${filePath}`);
      if (!dlRes.ok) return false;

      const buf = Buffer.from(await dlRes.arrayBuffer());
      fs.mkdirSync(path.dirname(mutable), { recursive: true });
      fs.writeFileSync(mutable, buf);
      try { fs.chmodSync(mutable, 0o644); } catch {}
      return true;
    } catch (err) {
      logger.warn({ err, agentId }, 'Failed to fetch avatar from Telegram');
      return false;
    }
  });
}

// ── Upload write path ────────────────────────────────────────────────
export interface UploadResult {
  ok: true;
  bytes: number;
  absPath: string;
  mtimeMs: number;
  size: number;
}

export async function writeUploadedAvatar(
  agentId: string,
  bytes: Buffer,
): Promise<UploadResult> {
  if (!ID_RE.test(agentId)) throw new Error('invalid agent id');

  const sig = bytes.subarray(0, 12);
  const isPng = sig[0] === 0x89 && sig[1] === 0x50 && sig[2] === 0x4e && sig[3] === 0x47;
  const isJpeg = sig[0] === 0xff && sig[1] === 0xd8 && sig[2] === 0xff;
  const isWebp = sig[0] === 0x52 && sig[1] === 0x49 && sig[2] === 0x46 && sig[3] === 0x46
              && sig[8] === 0x57 && sig[9] === 0x45 && sig[10] === 0x42 && sig[11] === 0x50;
  if (!isPng && !isJpeg && !isWebp) {
    throw new Error('image must be PNG, JPEG, or WebP');
  }

  return withAgentLock(agentId, async () => {
    const target = getMutableAvatarPath(agentId);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, bytes);
    try { fs.chmodSync(target, 0o644); } catch {}
    // Drop the sticky-fail flag so a fresh upload immediately serves
    // instead of waiting out the 24h "no Telegram photo" cooldown.
    const flag = noAvatarFlagPath(agentId);
    if (fs.existsSync(flag)) { try { fs.unlinkSync(flag); } catch {} }
    const st = fs.statSync(target);
    return { ok: true, bytes: bytes.length, absPath: target, mtimeMs: st.mtimeMs, size: st.size };
  });
}

export async function deleteUploadedAvatar(agentId: string): Promise<void> {
  if (!ID_RE.test(agentId)) throw new Error('invalid agent id');
  await withAgentLock(agentId, async () => {
    const target = getMutableAvatarPath(agentId);
    if (fs.existsSync(target)) fs.unlinkSync(target);
  });
}

// no-op kept for backwards compat with src/index.ts call site
export function runWarroomAvatarMigration(): void {}
