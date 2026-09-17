/**
 * Portfolio AI assistant — Cloudflare Pages Function.
 *
 * Deploy target: Cloudflare Pages (free tier), connected to this repo.
 *   - Static site is served from the repo root.
 *   - This file becomes POST /api/chat automatically.
 *   - The page finds it on its own (see CONFIG.chatEndpoint in index.html),
 *     and silently falls back to the built-in offline answers if it 404s.
 *
 * Uses Cloudflare Workers AI, so NO API key is needed — just bind it:
 *   Cloudflare dashboard -> your Pages project -> Settings -> Functions
 *   -> Workers AI binding -> variable name: AI
 *
 * Optional environment variables:
 *   ALLOWED_ORIGIN  extra allowed origin, e.g. https://udaym001.github.io
 *   GROQ_API_KEY    if set, uses Groq's OpenAI-compatible API instead of Workers AI
 *   OPENAI_API_KEY  if set, uses OpenAI instead (GROQ wins if both are set)
 *   AI_MODEL        override the model id
 */

const DEFAULT_WORKERS_AI_MODEL = '@cf/meta/llama-3.1-8b-instruct';
const DEFAULT_GROQ_MODEL = 'llama-3.1-8b-instant';
const DEFAULT_OPENAI_MODEL = 'gpt-4o-mini';

const MAX_MESSAGE_CHARS = 700;   // per visitor message
const MAX_MESSAGES = 10;         // conversation turns forwarded to the model
const MAX_OUTPUT_TOKENS = 400;

// Best-effort in-memory throttle (per isolate). Stops casual abuse.
const HITS = new Map();
const WINDOW_MS = 60_000;
const MAX_PER_WINDOW = 12;

function rateLimited(ip) {
  const now = Date.now();
  const entry = HITS.get(ip) || { count: 0, reset: now + WINDOW_MS };
  if (now > entry.reset) { entry.count = 0; entry.reset = now + WINDOW_MS; }
  entry.count += 1;
  HITS.set(ip, entry);
  if (HITS.size > 5000) for (const [k, v] of HITS) if (now > v.reset) HITS.delete(k);
  return entry.count > MAX_PER_WINDOW;
}

function corsHeaders(request, env) {
  const origin = request.headers.get('Origin') || '';
  const host = new URL(request.url).origin;
  const allowed = env.ALLOWED_ORIGIN || '';
  // Same-origin always fine; the GitHub Pages copy is allowed when configured.
  const ok = !origin || origin === host || (allowed && origin === allowed);
  return {
    'Access-Control-Allow-Origin': ok ? (origin || host) : host,
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
    Vary: 'Origin'
  };
}

const json = (data, status, extra = {}) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8', ...extra }
  });

/** Builds the system prompt from whatever context the page sent, with sane fallbacks. */
function buildSystemPrompt(context) {
  const name = context?.name || 'Manchu Uday Kiran';
  const email = context?.email || 'udaymanchu001@gmail.com';
  const resume = context?.resume || 'pdfs/UDAY.pdf';

  const projects = Array.isArray(context?.projects) && context.projects.length
    ? context.projects
        .map(p => `- ${p.title}: ${p.desc} (Stack: ${(p.tags || []).join(', ')}. Link: ${p.link})`)
        .join('\n')
    : '- (project list unavailable)';

  return [
    `You are the portfolio assistant for ${name}, a prompt engineer and automation developer.`,
    'You speak on his behalf to recruiters, clients and collaborators.',
    '',
    'Facts about him:',
    '- He builds AI-driven POCs and automation tools using Python, Playwright, LLM APIs, C#/.NET and WinForms.',
    '- He completed a Bachelor of Computer Applications at Vignan Degree & PG College.',
    '- He specialises in turning manual, repetitive workflows into fast automated systems.',
    `- Contact email: ${email}`,
    `- Resume: ${resume} (also linked from the header button)`,
    '',
    'Projects:',
    projects,
    '',
    'Rules:',
    `- Keep answers under 90 words unless asked for detail. Be warm, direct and specific.`,
    '- Only discuss his work, skills, background, availability and how to contact him.',
    '- Never invent projects, employers, dates or metrics that are not listed above.',
    `- If asked something you do not know, say so and suggest emailing ${email}.`,
    '- When useful, point people to the relevant section of the page (#projects, #contact).',
    '- Reply in plain text or light markdown (**bold**, bullet lists, [text](url)). No HTML.'
  ].join('\n');
}

/** Cleans + bounds the conversation the client sent. */
function normaliseMessages(raw) {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter(m => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
    .slice(-MAX_MESSAGES)
    .map(m => ({
      role: m.role,
      content: m.content.slice(0, MAX_MESSAGE_CHARS).replace(/\s+/g, ' ').trim()
    }))
    .filter(m => m.content.length > 0);
}

async function callWorkersAI(env, messages, system) {
  const model = env.AI_MODEL || DEFAULT_WORKERS_AI_MODEL;
  const res = await env.AI.run(model, {
    messages: [{ role: 'system', content: system }, ...messages],
    max_tokens: MAX_OUTPUT_TOKENS,
    temperature: 0.6
  });
  // Workers AI returns { response } for text models.
  return (res && (res.response || res.result?.response)) || '';
}

async function callOpenAICompatible(url, key, model, messages, system) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
    body: JSON.stringify({
      model,
      messages: [{ role: 'system', content: system }, ...messages],
      max_tokens: MAX_OUTPUT_TOKENS,
      temperature: 0.6
    })
  });
  if (!res.ok) throw new Error(`Upstream ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const data = await res.json();
  return data?.choices?.[0]?.message?.content || '';
}

export async function onRequestOptions({ request, env }) {
  return new Response(null, { status: 204, headers: corsHeaders(request, env) });
}

export async function onRequestGet({ request, env }) {
  const headers = corsHeaders(request, env);
  const provider = env.GROQ_API_KEY ? 'groq' : env.OPENAI_API_KEY ? 'openai' : env.AI ? 'workers-ai' : null;
  return json({ ok: true, ai: Boolean(provider), provider }, 200, headers);
}

export async function onRequestPost({ request, env }) {
  const headers = corsHeaders(request, env);

  const provider = env.GROQ_API_KEY ? 'groq' : env.OPENAI_API_KEY ? 'openai' : env.AI ? 'workers-ai' : null;
  if (!provider) {
    // The page treats this as "no backend" and uses its offline answers.
    return json({ ok: false, error: 'No AI binding configured' }, 503, headers);
  }

  const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
  if (rateLimited(ip)) {
    return json({ ok: false, error: 'Too many requests, slow down.' }, 429, headers);
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return json({ ok: false, error: 'Invalid JSON body' }, 400, headers);
  }

  const messages = normaliseMessages(body?.messages);
  if (!messages.length) {
    return json({ ok: false, error: 'No message provided' }, 400, headers);
  }

  const system = buildSystemPrompt(body?.context);

  try {
    let reply;
    if (provider === 'groq') {
      reply = await callOpenAICompatible(
        'https://api.groq.com/openai/v1/chat/completions',
        env.GROQ_API_KEY,
        env.AI_MODEL || DEFAULT_GROQ_MODEL,
        messages, system
      );
    } else if (provider === 'openai') {
      reply = await callOpenAICompatible(
        'https://api.openai.com/v1/chat/completions',
        env.OPENAI_API_KEY,
        env.AI_MODEL || DEFAULT_OPENAI_MODEL,
        messages, system
      );
    } else {
      reply = await callWorkersAI(env, messages, system);
    }

    reply = String(reply || '').trim();
    if (!reply) return json({ ok: false, error: 'Empty model response' }, 502, headers);

    return json({ ok: true, reply, provider }, 200, headers);
  } catch (err) {
    // Never leak upstream detail to the browser; the page falls back offline.
    console.error('chat error:', err);
    return json({ ok: false, error: 'Upstream failure' }, 502, headers);
  }
}
