import fs from 'node:fs/promises';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';

const experimentRoot = path.dirname(new URL(import.meta.url).pathname.replace(/^\/(.:)/, '$1'));
const variant = process.argv[2];

if (!['stable', 'alpha'].includes(variant)) {
  throw new Error('Usage: node run-fixtures.mjs <stable|alpha>');
}

const moduleRoot = path.join(experimentRoot, variant, 'node_modules');
const layoutModule = await import(pathToFileURL(path.join(moduleRoot, 'bpmn-auto-layout', 'dist', 'index.js')));
const moddleModule = await import(pathToFileURL(path.join(moduleRoot, 'bpmn-moddle', 'dist', 'index.js')));
const { layoutProcess } = layoutModule;
const BpmnModdle = moddleModule.BpmnModdle;
const moddle = new BpmnModdle();
const inputRoot = path.join(experimentRoot, 'inputs');
const outputRoot = path.join(experimentRoot, 'outputs', variant);
const fixtureNames = [
  'linear_process',
  'exclusive_gateway',
  'parallel_split_join',
  'loop_corrections',
  'two_participants_message',
  'nested_subprocess',
  'nested_subprocess_expanded',
  'complex_approval',
  'complex_collaboration',
  'complex_nested_process'
];

await fs.mkdir(outputRoot, { recursive: true });

function collectSemantics(definitions) {
  const elements = [];
  const connections = [];
  const visitFlowElements = (flowElements = []) => {
    for (const element of flowElements) {
      elements.push([element.id, element.$type]);
      if (element.$type === 'bpmn:SequenceFlow') {
        connections.push([element.id, element.$type, element.sourceRef?.id, element.targetRef?.id]);
      }
      if (element.$type === 'bpmn:SubProcess') {
        visitFlowElements(element.flowElements);
      }
    }
  };
  for (const root of definitions.rootElements || []) {
    elements.push([root.id, root.$type]);
    if (root.$type === 'bpmn:Process') {
      visitFlowElements(root.flowElements);
    }
    if (root.$type === 'bpmn:Collaboration') {
      for (const participant of root.participants || []) {
        elements.push([participant.id, participant.$type]);
        connections.push([participant.id, participant.$type, participant.processRef?.id, null]);
      }
      for (const flow of root.messageFlows || []) {
        elements.push([flow.id, flow.$type]);
        connections.push([flow.id, flow.$type, flow.sourceRef?.id, flow.targetRef?.id]);
      }
    }
  }
  return {
    elements: elements.sort((a, b) => a[0].localeCompare(b[0])),
    connections: connections.sort((a, b) => a[0].localeCompare(b[0]))
  };
}

function collectExpectedVisuals(definitions) {
  const shapes = new Set();
  const edges = new Set();
  const realIds = new Set();
  const visitLanes = (laneSets = []) => {
    for (const laneSet of laneSets) {
      realIds.add(laneSet.id);
      for (const lane of laneSet.lanes || []) {
        realIds.add(lane.id);
        shapes.add(lane.id);
        visitLanes(lane.childLaneSet ? [lane.childLaneSet] : []);
      }
    }
  };
  const visitFlowElements = (flowElements = []) => {
    for (const element of flowElements) {
      realIds.add(element.id);
      if (element.$type === 'bpmn:SequenceFlow') {
        edges.add(element.id);
      } else {
        shapes.add(element.id);
      }
      if (element.$type === 'bpmn:SubProcess') {
        visitLanes(element.laneSets);
        visitFlowElements(element.flowElements);
      }
    }
  };
  for (const root of definitions.rootElements || []) {
    realIds.add(root.id);
    if (root.$type === 'bpmn:Process') {
      visitLanes(root.laneSets);
      visitFlowElements(root.flowElements);
    }
    if (root.$type === 'bpmn:Collaboration') {
      for (const participant of root.participants || []) {
        realIds.add(participant.id);
        shapes.add(participant.id);
      }
      for (const flow of root.messageFlows || []) {
        realIds.add(flow.id);
        edges.add(flow.id);
      }
    }
  }
  return { shapes, edges, realIds };
}

function countModel(definitions) {
  let nodeCount = 0;
  let sequenceFlowCount = 0;
  let messageFlowCount = 0;
  const visit = (flowElements = []) => {
    for (const element of flowElements) {
      if (element.$type === 'bpmn:SequenceFlow') {
        sequenceFlowCount += 1;
      } else {
        nodeCount += 1;
      }
      if (element.$type === 'bpmn:SubProcess') {
        visit(element.flowElements);
      }
    }
  };
  for (const root of definitions.rootElements || []) {
    if (root.$type === 'bpmn:Process') {
      visit(root.flowElements);
    } else if (root.$type === 'bpmn:Collaboration') {
      messageFlowCount += (root.messageFlows || []).length;
    }
  }
  return { nodeCount, sequenceFlowCount, messageFlowCount, flowCount: sequenceFlowCount + messageFlowCount };
}

function inspectDi(definitions, expected) {
  const errors = [];
  const shapeIds = new Set();
  const edgeIds = new Set();
  const planes = [];
  let subprocessExpanded = null;
  for (const diagram of definitions.diagrams || []) {
    const plane = diagram.plane;
    if (!plane) {
      errors.push(`Diagram ${diagram.id} has no BPMNPlane`);
      continue;
    }
    planes.push({ id: plane.id, bpmnElement: plane.bpmnElement?.id });
    if (plane.bpmnElement?.id && !expected.realIds.has(plane.bpmnElement.id)) {
      errors.push(`Plane ${plane.id} references unknown ${plane.bpmnElement.id}`);
    }
    for (const item of plane.planeElement || []) {
      const semanticId = item.bpmnElement?.id;
      if (!semanticId || !expected.realIds.has(semanticId)) {
        errors.push(`${item.id} references unknown BPMN element ${semanticId}`);
      }
      if (item.$type === 'bpmndi:BPMNShape') {
        shapeIds.add(semanticId);
        const bounds = item.bounds;
        if (!bounds || ![bounds.x, bounds.y, bounds.width, bounds.height].every(Number.isFinite) || bounds.width <= 0 || bounds.height <= 0) {
          errors.push(`Shape ${semanticId} has invalid bounds`);
        }
        if (semanticId === 'verification') {
          subprocessExpanded = item.isExpanded === true;
        }
      } else if (item.$type === 'bpmndi:BPMNEdge') {
        edgeIds.add(semanticId);
        const waypoints = item.waypoint || [];
        if (waypoints.length < 2 || !waypoints.every(point => Number.isFinite(point.x) && Number.isFinite(point.y))) {
          errors.push(`Edge ${semanticId} has invalid waypoints`);
        }
      }
    }
  }
  const missingShapes = [...expected.shapes].filter(id => !shapeIds.has(id)).sort();
  const missingEdges = [...expected.edges].filter(id => !edgeIds.has(id)).sort();
  return { errors, missingShapes, missingEdges, planes, subprocessExpanded, shapeCount: shapeIds.size, edgeCount: edgeIds.size };
}

const report = { variant, packageVersion: variant === 'stable' ? '1.3.0' : '2.0.0-alpha.2', fixtures: {} };

for (const fixtureName of fixtureNames) {
  const inputXml = await fs.readFile(path.join(inputRoot, `${fixtureName}.bpmn`), 'utf8');
  try {
    const inputParsed = await moddle.fromXML(inputXml);
    const before = collectSemantics(inputParsed.rootElement);
    const counts = countModel(inputParsed.rootElement);
    const layoutStarted = performance.now();
    const layoutResult = await layoutProcess(inputXml);
    const layoutDurationMs = performance.now() - layoutStarted;
    const outputXml = typeof layoutResult === 'string' ? layoutResult : layoutResult.xml;
    const warnings = typeof layoutResult === 'string' ? [] : (layoutResult.warnings || []).map(warning => ({ code: warning.code, message: warning.message }));
    const outputParsed = await moddle.fromXML(outputXml);
    const after = collectSemantics(outputParsed.rootElement);
    const expected = collectExpectedVisuals(inputParsed.rootElement);
    const di = inspectDi(outputParsed.rootElement, expected);
    const semanticsPreserved = JSON.stringify(before) === JSON.stringify(after);
    if (!semanticsPreserved) {
      di.errors.push('Semantic elements or references changed');
    }
    const status = di.errors.length || di.missingShapes.length || di.missingEdges.length ? 'PARTIAL' : 'PASS';
    report.fixtures[fixtureName] = { status, warnings, semanticsPreserved, ...counts, layoutDurationMs, ...di };
    await fs.writeFile(path.join(outputRoot, `${fixtureName}.bpmn`), outputXml, 'utf8');
  } catch (error) {
    report.fixtures[fixtureName] = {
      status: 'FAIL',
      errorName: error.name,
      errorCode: error.code,
      errorMessage: error.message,
      stack: error.stack
    };
  }
}

await fs.writeFile(path.join(outputRoot, 'report.json'), JSON.stringify(report, null, 2), 'utf8');
console.log(JSON.stringify(report, null, 2));
