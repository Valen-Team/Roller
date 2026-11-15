// Observe the native adapter's serialized fetch body; never record headers.
import { appendFileSync, readFileSync, writeFileSync } from 'node:fs';

export const name = 'roller-request-journal';
// The sdk profile's delegation tools. Children would share this process and
// its journal, which records one linear conversation, and the shipped agent
// preset cannot be patched, so the tools are withheld from the model instead.
const DELEGATION_TOOLS = new Set(['subagent', 'subagent_fork', 'list_agents', 'interrupt_agent',
  'send_message', 'workflow', 'ralph']);
export const inject = ['subprocess'];
export function apply(ctx) {
  const original = globalThis.fetch;
  let turn, count = 0;
  const journal = process.env.ROLLER_DEEPSEEK_JOURNAL;
  // dsh scrubs credential-shaped ambient names. Restore only environment
  // entries the Roller task or --pass-env explicitly requested, using the
  // native subprocess service's deliberate env override seam.
  // sdk-minimal runs tools in its terminal; other profiles also spawn plain
  // subprocesses.
  const service = ctx.subprocess;
  const {profile} = JSON.parse(readFileSync('/roller/deepseek-state/turn.json', 'utf8'));
  const originals = new Map();
  for (const method of profile === 'sdk-minimal' ? ['spawnTerminal'] : ['spawn', 'spawnTerminal']) {
    const original = service[method];
    originals.set(method, original);
    service[method] = function (spec) {
      const config = JSON.parse(readFileSync('/roller/deepseek-state/turn.json', 'utf8'));
      const env = Object.fromEntries(config.tool_env_names.filter(n => process.env[n] !== undefined)
        .map(n => [n, process.env[n]]));
      return original.call(service, {...spec, env: {...env, ...spec.env}});
    };
  }
  globalThis.fetch = async function (input, init) {
    const url = typeof input === 'string' ? input : input.url ?? String(input);
    if (new URL(url).pathname.endsWith('/chat/completions')) {
      const config = JSON.parse(readFileSync('/roller/deepseek-state/turn.json', 'utf8'));
      if (turn !== config.turn) { turn = config.turn; count = 0; }
      if (++count > config.max_steps) {
        const error = `Roller DeepSeek max steps exceeded (${config.max_steps})`;
        appendFileSync(journal, JSON.stringify({type: 'runner_error', turn, error}) + '\n');
        writeFileSync('/roller/deepseek-state/limit.json', JSON.stringify({turn, error}));
        throw new Error(error);
      }
      if (typeof init?.body !== 'string') throw new Error('DeepSeek request body is not serialized JSON');
      const body = JSON.parse(init.body);
      if (Array.isArray(body.tools) && body.tools.some(tool => DELEGATION_TOOLS.has(tool.function?.name))) {
        body.tools = body.tools.filter(tool => !DELEGATION_TOOLS.has(tool.function?.name));
        init = {...init, body: JSON.stringify(body)};
      }
      appendFileSync(journal, JSON.stringify({type: 'request', turn, body}) + '\n');
      if (config.capture_calls) {
        const headers = new Headers(init.headers ?? (typeof input === 'object' ? input.headers : undefined));
        headers.set('x-roller-native-session-id', config.session_id);
        init = {...init, headers};
      }
    }
    return original.call(globalThis, input, init);
  };
  ctx.on('dispose', () => { globalThis.fetch = original; for (const [method, original] of originals) service[method] = original; });
}
