# Page topology

The default Vite route is the reproduction progress report. `?view=official`
links to the authors' reference simulation at its official host.
`?view=trained` reads `models/releases.json`, then loads the active audited
export with browser-only ONNX and MuJoCo workers. An empty registry explicitly
shows that no model has been released.

The independent, procedural illustration at `?native=1` is optional and does
not run policy inference, MuJoCo, or any copied demo assets:

1. **WebGL stage**
   - Fixed full-viewport canvas with transparent renderer.
   - Pale radial-gradient body background, receiving floor shadow, fading grid, and centered robot scene.
   - Interaction model is time-driven worker animation plus pointer drag and orbit camera.

2. **Legend panel**
   - Fixed top-left panel. Desktop includes the `Live imagination` heading. Embedded/mobile layout keeps the segmented legend visible.
   - World model key, Goal toggle, and MuJoCo toggle.
   - Step counter sits below the segmented control.

3. **SuperMimic link**
   - Fixed top-right glass pill linking back to the project site.

4. **Contextual hint and splash**
   - Hint chip is absolutely positioned over the scene and follows the robot projection.
   - Splash owns the viewport during its short headline state and fades surrounding chrome.

5. **Bottom dock**
   - Fixed bottom-center glass pill. Desktop includes reset, play/pause, policy switch, action-noise slider, friction slider, and motions selector.
   - Mobile collapses controls into a sheet trigger and retains reset, play/pause, and motions.

6. **Motions sheet**
   - Desktop popover above the dock's right edge. Mobile fixed sheet above the dock.
   - Search field, category folders, motion items, and license footer.

7. **Boot card**
   - Centered dense glass panel shown during model setup, removed after first state arrives.
