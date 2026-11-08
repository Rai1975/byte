// Presage SmartSpectra talking detector, driven by frames piped in from byte/app.py.
//
// Python owns the webcam (Windows only lets one process open it), so frames come in on stdin:
//   header (16 bytes, little endian): u32 width, u32 height, f64 timestampUs
//   payload: width * height * 3 bytes, BGR
// Results go out on stdout as one JSON object per line:
//   {"type":"status","status":3}                              ProcessingStatus (3 = running)
//   {"type":"talking","detected":true,"stable":true,"ts":...} Presage TALKING metric
//   {"type":"validation","code":..,"hint":".."}               e.g. face not visible
//   {"type":"error","code":..,"message":"..","retryable":false}
//
// Usage: PRESAGE_API_KEY=... node talk_bridge.mjs [--debug]

import { SmartSpectraSDK, PixelFormat, MetricType } from '@smartspectra/node-sdk';
import { decodeMetrics } from '@smartspectra/node-sdk/messages';

const DEBUG = process.argv.includes('--debug');
const HEADER = 16;

const emit = (obj) => process.stdout.write(JSON.stringify(obj) + '\n');
const log = (...a) => { if (DEBUG) console.error('[bridge]', ...a); };
// stdout to a pipe is async on Windows: exit only after the last message is flushed
const fail = (obj, code = 1) => process.stdout.write(JSON.stringify({ type: 'error', ...obj }) + '\n', () => process.exit(code));

const apiKey = process.env.PRESAGE_API_KEY;
if (!apiKey) {
  fail({ code: -1, message: 'PRESAGE_API_KEY not set', retryable: false }, 2);
}

const sdk = new SmartSpectraSDK({
  apiKey,
  requestedMetrics: [MetricType.TALKING],
  enableTelemetry: false,
});

let shuttingDown = false;
let started = false; // send frames as soon as start() succeeds; with custom input the SDK may wait for frames before reporting "running"
let lastTs = 0;

let wasRunning = false;
let lastErrorRetryable = false;
let restarting = false;
const restartTimes = [];
const MAX_RESTARTS_PER_MIN = 5;

// Retryable errors (e.g. code 8 "processing failed") put the session in kError (5). Rebuild it
// with reset() + start() instead of giving up; give up only if it keeps failing.
function restart() {
  const now = Date.now();
  while (restartTimes.length && now - restartTimes[0] > 60000) restartTimes.shift();
  if (restartTimes.length >= MAX_RESTARTS_PER_MIN) {
    fail({ code: -3, retryable: false, message: `Presage kept failing (${MAX_RESTARTS_PER_MIN} restarts in 1 min)` });
    return;
  }
  restartTimes.push(now);
  restarting = true;
  started = false;
  emit({ type: 'restart', count: restartTimes.length });
  setTimeout(() => {
    try {
      sdk.reset();
      sdk.useCustomInput();
      sdk.start();
      started = true;
    } catch (e) {
      fail({ code: e.code ?? -1, message: 'restart failed: ' + e.message, retryable: false });
    } finally {
      restarting = false;
    }
  }, 300);
}

sdk.on('processingStatus', (status) => {
  emit({ type: 'status', status });
  if (status === 3) wasRunning = true;
  if (shuttingDown || restarting) return;
  if (status === 5) {
    if (lastErrorRetryable) restart();
    else process.stdout.write('', () => process.exit(1));   // terminal (reported via 'error')
    return;
  }
  // A bad key doesn't raise 'error': the session just drops back to idle (status 1) and the
  // SDK logs "Usage verification failed. Status code: 401" to stderr.
  if (status === 1 && started) {
    fail({ code: -2, retryable: false, message: wasRunning
      ? 'Presage session stopped unexpectedly'
      : 'Presage session stopped during startup (API key rejected?)' });
  }
});
sdk.on('validationStatus', (code, ts, hint) => emit({ type: 'validation', code, hint }));
sdk.on('error', (code, message, retryable) => {
  lastErrorRetryable = !!retryable;
  emit({ type: 'error', code, message, retryable });
});
sdk.on('metrics', (buf) => {
  const m = decodeMetrics(buf);
  const talking = m?.face?.talking;
  if (DEBUG) log('metrics', JSON.stringify(m));
  if (talking && talking.length) {
    const last = talking[talking.length - 1];
    emit({ type: 'talking', detected: !!last.detected, stable: !!last.stable, ts: Number(last.timestamp ?? 0) });
  }
});

try {
  sdk.useCustomInput();
  sdk.start();
  started = true;
  // start() can return "successfully" with the session already back to idle/error
  // (e.g. the key was rejected during startup), so check where it actually ended up.
  const st = sdk.processingStatus;
  if (st === 1 || st === 5) {
    fail({ code: -2, retryable: false, message: 'Presage session did not start (API key rejected?)' });
  }
} catch (e) {
  fail({ code: e.code ?? -1, message: e.message, retryable: !!e.retryable });
}

// ---- stdin frame reader -------------------------------------------------------------------
let chunks = [];
let have = 0;

function take(n) {
  const all = chunks.length === 1 ? chunks[0] : Buffer.concat(chunks, have);
  const out = all.subarray(0, n);
  const rest = all.subarray(n);
  chunks = rest.length ? [rest] : [];
  have = rest.length;
  return out;
}

let pending = null; // parsed header waiting for its payload

process.stdin.on('data', (chunk) => {
  chunks.push(chunk);
  have += chunk.length;
  for (;;) {
    if (!pending) {
      if (have < HEADER) return;
      const h = take(HEADER);
      pending = { w: h.readUInt32LE(0), h: h.readUInt32LE(4), ts: h.readDoubleLE(8) };
    }
    const size = pending.w * pending.h * 3;
    if (have < size) return;
    const frame = take(size);
    const { w, h, ts } = pending;
    pending = null;
    if (!started || ts <= lastTs) continue;
    lastTs = ts;
    try {
      sdk.sendFrame(frame, w, h, w * 3, PixelFormat.kBGR, Math.round(ts));
    } catch (e) {
      log('sendFrame failed:', e.message);
    }
  }
});

async function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  try { await sdk.stopAsync(); } catch {}
  try { await sdk.destroy(); } catch {}
  process.exit(0);
}
process.stdin.on('end', shutdown);
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
