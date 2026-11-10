# BPMN auto-layout feasibility and visual-validation experiment

This isolated experiment compares the npm `latest` release (`bpmn-auto-layout@1.3.0`) with the npm `next` prerelease (`bpmn-auto-layout@2.0.0-alpha.2`). It does not modify or wrap production modules.

Generate semantic inputs from the six repository fixtures and three stress
scenarios. The generator validates each JSON fixture as a `ProcessModel` and
uses the production `serialize_bpmn()` function; semantic BPMN is not authored
by hand.

```powershell
.venv\Scripts\python.exe experiments\bpmn-auto-layout-feasibility\generate_inputs.py
```

Install the pinned packages locally:

```powershell
Set-Location experiments\bpmn-auto-layout-feasibility\stable
npm.cmd install --ignore-scripts
Set-Location ..\alpha
npm.cmd install --ignore-scripts
```

Run the common structural checker from the repository root. Mission 004-B uses
only the pinned alpha package; the stable commands remain solely to reproduce
the earlier Mission 004-A comparison.

```powershell
node experiments\bpmn-auto-layout-feasibility\run-fixtures.mjs stable
node experiments\bpmn-auto-layout-feasibility\run-fixtures.mjs alpha
```

Build and run the local `bpmn-js` browser validation with the installed
Microsoft Edge browser (no browser download is required):

```powershell
Set-Location experiments\bpmn-auto-layout-feasibility\alpha
node build-browser.mjs
node run-visual.mjs
```

The browser run imports each alpha-layouted XML document into pinned
`bpmn-js@18.31.1`, checks rendered element IDs and import/console errors, and
captures screenshots in `outputs/visual/`. Browser launch and BPMN import have
15-second and 10-second limits respectively.

Generated inputs, outputs, reports, and `node_modules` are intentionally ignored. Package manifests and lockfiles preserve the tested versions.

The structural checker verifies semantic identity with `bpmn-moddle`, DI
references, finite positive shape bounds, and edges with at least two finite
waypoints. The browser harness verifies real local rendering, but neither
harness performs official BPMN XSD validation or establishes full BPMN 2.0
conformance.

Only collapsed subprocess rendering is supported by this prototype. Expanded
subprocess input must be rejected by a future adapter; it must not be silently
converted to collapsed form. The alpha layout engine currently returns a
collapsed subprocess even when the seed DI requests expansion.
