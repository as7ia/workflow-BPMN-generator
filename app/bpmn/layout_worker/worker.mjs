import { layoutProcess } from 'bpmn-auto-layout';

const chunks = [];
for await (const chunk of process.stdin) {
  chunks.push(chunk);
}

const inputXml = Buffer.concat(chunks).toString('utf8');

try {
  const result = await layoutProcess(inputXml);
  const xml = typeof result === 'string' ? result : result.xml;
  const warnings = typeof result === 'string' ? [] : (result.warnings ?? []).map((warning) => ({
    code: warning.code ?? null,
    message: warning.message ?? String(warning)
  }));
  process.stdout.write(JSON.stringify({ xml, warnings }));
} catch (error) {
  process.stderr.write(`${error?.stack ?? error}\n`);
  process.exitCode = 1;
}
