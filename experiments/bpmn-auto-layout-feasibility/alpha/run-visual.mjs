import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { BpmnModdle } from 'bpmn-moddle';
import puppeteer from 'puppeteer-core';

const experimentRoot = path.resolve('..');
const outputRoot = path.join(experimentRoot, 'outputs');
const alphaOutput = path.join(outputRoot, 'alpha');
const visualOutput = path.join(outputRoot, 'visual');
const edgeExecutable = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const fixtureNames = [
  'linear_process',
  'exclusive_gateway',
  'parallel_split_join',
  'loop_corrections',
  'two_participants_message',
  'nested_subprocess',
  'complex_approval',
  'complex_collaboration',
  'complex_nested_process'
];
const moddle = new BpmnModdle();

await fs.mkdir(visualOutput, { recursive: true });

const browser = await puppeteer.launch({
  executablePath: edgeExecutable,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu'],
  timeout: 15000
});

const report = {
  browser: 'Microsoft Edge 157.0.4341.42',
  bpmnJs: '18.31.1',
  fixtures: {}
};

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1800, height: 1100, deviceScaleFactor: 1 });
  await page.goto(pathToFileURL(path.resolve('browser.html')).href, {
    waitUntil: 'load',
    timeout: 10000
  });

  let activeFixture = null;
  const consoleErrors = new Map();
  const pageErrors = new Map();
  page.on('console', message => {
    if (activeFixture && message.type() === 'error') {
      consoleErrors.get(activeFixture).push(message.text());
    }
  });
  page.on('pageerror', error => {
    if (activeFixture) {
      pageErrors.get(activeFixture).push(error.message);
    }
  });

  for (const fixtureName of fixtureNames) {
    activeFixture = fixtureName;
    consoleErrors.set(fixtureName, []);
    pageErrors.set(fixtureName, []);
    const xml = await fs.readFile(path.join(alphaOutput, `${fixtureName}.bpmn`), 'utf8');
    const parsed = await moddle.fromXML(xml);
    const firstPlane = parsed.rootElement.diagrams?.[0]?.plane;
    const expectedIds = (firstPlane?.planeElement || [])
      .map(element => element.bpmnElement?.id)
      .filter(Boolean)
      .sort();
    const expectedShapeIds = (firstPlane?.planeElement || [])
      .filter(element => element.$type === 'bpmndi:BPMNShape')
      .map(element => element.bpmnElement?.id)
      .filter(Boolean)
      .sort();
    const expectedEdgeIds = (firstPlane?.planeElement || [])
      .filter(element => element.$type === 'bpmndi:BPMNEdge')
      .map(element => element.bpmnElement?.id)
      .filter(Boolean)
      .sort();

    const renderResult = await page.evaluate(async inputXml => {
      return Promise.race([
        window.renderBpmn(inputXml),
        new Promise((_, reject) => setTimeout(() => reject(new Error('bpmn-js import timed out after 10000 ms')), 10000))
      ]);
    }, xml);
    await new Promise(resolve => setTimeout(resolve, 100));

    const renderedIds = renderResult.elements.map(element => element.id).sort();
    const missingRenderedIds = expectedIds.filter(id => !renderedIds.includes(id));
    const renderedShapeIds = renderResult.elements
      .filter(element => !element.isConnection)
      .map(element => element.id);
    const renderedEdgeIds = renderResult.elements
      .filter(element => element.isConnection)
      .map(element => element.id);
    const missingRenderedShapes = expectedShapeIds.filter(id => !renderedShapeIds.includes(id));
    const missingRenderedEdges = expectedEdgeIds.filter(id => !renderedEdgeIds.includes(id));
    const screenshot = path.join(visualOutput, `${fixtureName}.png`);
    await page.screenshot({ path: screenshot, fullPage: false });

    const fatalErrors = [...consoleErrors.get(fixtureName), ...pageErrors.get(fixtureName)];
    report.fixtures[fixtureName] = {
      status: fatalErrors.length || renderResult.warnings.length || missingRenderedIds.length ? 'FAIL' : 'PASS',
      importDurationMs: renderResult.importDurationMs,
      importWarnings: renderResult.warnings,
      consoleErrors: consoleErrors.get(fixtureName),
      pageErrors: pageErrors.get(fixtureName),
      expectedShapeCount: expectedShapeIds.length,
      expectedEdgeCount: expectedEdgeIds.length,
      renderedShapeCount: renderResult.renderedShapeCount,
      renderedConnectionCount: renderResult.renderedConnectionCount,
      missingRenderedShapes,
      missingRenderedEdges,
      svgCount: renderResult.svgCount,
      subprocesses: renderResult.subprocesses,
      screenshot
    };
  }
} finally {
  await browser.close();
}

await fs.writeFile(
  path.join(visualOutput, 'report.json'),
  JSON.stringify(report, null, 2),
  'utf8'
);
console.log(JSON.stringify(report, null, 2));
