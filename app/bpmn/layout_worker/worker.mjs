import { layoutProcess } from 'bpmn-auto-layout';

const chunks = [];
for await (const chunk of process.stdin) {
  chunks.push(chunk);
}

const inputXml = Buffer.concat(chunks).toString('utf8');

try {
  const result = await layoutProcess(inputXml);
  const xml = typeof result === 'string' ? result : result.xml;
  const maxOutputBytes = Number.parseInt(process.env.BPMN_LAYOUT_MAX_OUTPUT_BYTES ?? '', 10);
  if (!Number.isSafeInteger(maxOutputBytes) || maxOutputBytes <= 0) {
    throw new Error('BPMN_LAYOUT_MAX_OUTPUT_BYTES must be a positive integer');
  }
  const outputBytes = Buffer.byteLength(xml, 'utf8');
  if (outputBytes > maxOutputBytes) {
    throw new Error(`Layout output exceeds the configured ${maxOutputBytes}-byte limit`);
  }
  const warnings = typeof result === 'string' ? [] : (result.warnings ?? []).map((warning) => ({
    code: warning.code ?? null,
    message: warning.message ?? String(warning)
  }));
  process.stdout.write(JSON.stringify({ xml, warnings }));
} catch (error) {
  process.stderr.write(`${error?.stack ?? error}\n`);
  process.exitCode = 1;
}
