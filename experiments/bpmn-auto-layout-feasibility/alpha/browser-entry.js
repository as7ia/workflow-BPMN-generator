import BpmnViewer from 'bpmn-js/lib/NavigatedViewer';
import 'bpmn-js/dist/assets/diagram-js.css';

let viewer;

window.renderBpmn = async function renderBpmn(xml) {
  if (viewer) {
    viewer.destroy();
  }
  const canvasContainer = document.querySelector('#canvas');
  canvasContainer.innerHTML = '';
  viewer = new BpmnViewer({ container: canvasContainer });
  const started = performance.now();
  const result = await viewer.importXML(xml);
  const importDurationMs = performance.now() - started;
  const canvas = viewer.get('canvas');
  canvas.zoom('fit-viewport');
  const elementRegistry = viewer.get('elementRegistry');
  const elements = elementRegistry.getAll()
    .filter(element => element.businessObject?.id && !element.labelTarget)
    .map(element => ({
      id: element.businessObject.id,
      type: element.businessObject.$type,
      isConnection: Array.isArray(element.waypoints),
      x: element.x,
      y: element.y,
      width: element.width,
      height: element.height
    }));
  const subprocesses = elements
    .filter(element => element.type === 'bpmn:SubProcess')
    .map(element => {
      const registryElement = elementRegistry.get(element.id);
      return {
        id: element.id,
        isExpanded: registryElement.businessObject.di?.isExpanded === true
      };
    });
  return {
    importDurationMs,
    warnings: (result.warnings || []).map(warning => warning.message || String(warning)),
    elements,
    subprocesses,
    svgCount: canvasContainer.querySelectorAll('svg').length,
    renderedShapeCount: canvasContainer.querySelectorAll('.djs-shape').length,
    renderedConnectionCount: canvasContainer.querySelectorAll('.djs-connection').length
  };
};
