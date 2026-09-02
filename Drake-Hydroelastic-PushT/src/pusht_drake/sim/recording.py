"""Meshcat HTML replay recording.

StaticHtml() only serializes meshcat commands issued after StartRecording, so
every piece of static geometry (arm, table, T, goal) must be re-published by a
forced diagram publish once recording is live, otherwise the saved HTML
animates an empty scene. The goal markers are likewise re-drawn with their
time_in_recording so they exist inside the animation timeline.

One recorder per rig; one HTML per episode. Frames are captured
automatically: the station's visualizer publishes during AdvanceTo, and while
the meshcat instance is recording those transforms land in the animation.
This module imports no pydrake at module level, so the portability helper is
usable from the simulator-free check suite.
"""

from __future__ import annotations

from pathlib import Path

# 30 fps: the live Meshcat view publishes at 64 Hz; recording at the policy
# rate (10) sampled the arm's motion 6.4x more coarsely than the live view,
# which read as unreal playback. Frame density does not change playback
# duration (the animation carries sim timestamps).
RECORDING_FPS = 30.0

# A saved replay must open inside a sandboxed viewer, not only in a browser
# tab. Drake's meshcat.js gates its gamepad polling on a property test,
#
#     const gamepads_supported = !!navigator.getGamepads;
#
# which still passes under a restrictive Permissions-Policy because the
# property exists, and then calls it from inside the render loop. When the
# call throws SecurityError, the exception escapes before
# requestAnimationFrame, so the loop never reschedules: the page dies after a
# single frame and shows nothing. That blank scene needs no gamepad to
# reproduce, only an embedding that disallows the feature.
#
# Three independent defences, because a viewer cannot be inspected from here:
#   1. force the feature flag false, so the call is never made at all (this
#      is meshcat's own no-gamepad branch; its console signature, expected in
#      every saved replay in every browser, is "Warning: Gamepads are not
#      supported in this browser session.");
#   2. keep a shim that makes any surviving call return an empty list instead
#      of throwing;
#   3. a requestAnimationFrame fallback: some embedded webviews load and run
#      the page but never deliver rAF callbacks (no compositor vsync), and
#      meshcat drives both its render loop and its animation clock from rAF,
#      so the symptom is a scene that never appears or never plays, with no
#      console error at all. WebGL itself does not need the compositor, so a
#      timer-driven pump restores playback there; in a normal browser the
#      first native rAF tick disarms the fallback permanently.
# The bundled viewer is untouched; only the artifact written here is patched.
_GAMEPAD_FLAG = "const gamepads_supported = !!navigator.getGamepads;"
_GAMEPAD_FLAG_OFF = "const gamepads_supported = false;  /* disabled for sandboxed viewers */"

_GAMEPAD_SHIM = """<script>
(function () {
  var native = navigator.getGamepads && navigator.getGamepads.bind(navigator);
  if (!native) { return; }
  navigator.getGamepads = function () {
    try { return native(); } catch (error) { return []; }
  };
})();
</script>
"""

# Installed in the head so the bundle resolves the override at call time.
# Every requested callback lands in one queue; a perpetual native heartbeat
# pumps the queue while the compositor is delivering ticks, and a watchdog
# timer pumps it whenever native ticks go stale. Stall detection has to run
# continuously rather than as a one-shot probe at load: a surface can
# composite a few frames during load and stall afterwards, and a probe that
# saw those early ticks would disarm the fallback exactly when it is needed.
# Handoff works in both directions, repeatedly, so becoming visible again
# returns control to native pacing.
_RAF_FALLBACK = """<script>
/* raf fallback for viewers that stop delivering animation frames */
(function () {
  var native = window.requestAnimationFrame
    ? window.requestAnimationFrame.bind(window) : null;
  var pending = [];
  var lastNativeTick = 0;
  function pump() {
    var callbacks = pending;
    pending = [];
    for (var i = 0; i < callbacks.length; i++) {
      try { callbacks[i](performance.now()); } catch (error) { /* keep going */ }
    }
  }
  if (native) {
    (function heartbeat() {
      lastNativeTick = performance.now();
      pump();
      native(heartbeat);
    })();
  }
  setInterval(function () {
    if (pending.length > 0 && performance.now() - lastNativeTick > 500) {
      pump();
    }
  }, 33);
  window.requestAnimationFrame = function (callback) {
    pending.push(callback);
    return 0;
  };
})();
</script>
"""


def make_portable(html: str) -> str:
    """Make a Meshcat StaticHtml page survive a sandboxed viewer.

    Idempotent, and raises when the page is not the shape this expects: a
    no-op here would ship a replay that cannot be opened.
    """

    if "<head>" not in html:
        raise ValueError("unrecognized Meshcat HTML: no <head> to patch")
    patched = html
    if _GAMEPAD_FLAG in patched:
        patched = patched.replace(_GAMEPAD_FLAG, _GAMEPAD_FLAG_OFF, 1)
    elif _GAMEPAD_FLAG_OFF not in patched:
        raise ValueError(
            "unrecognized Meshcat HTML: the gamepad feature flag is not where "
            "this Drake version put it; the replay would die on the first frame "
            "inside a sandboxed viewer"
        )
    if "navigator.getGamepads = function" not in patched:
        patched = patched.replace("<head>", "<head>\n" + _GAMEPAD_SHIM, 1)
    if "raf fallback" not in patched:
        patched = patched.replace("<head>", "<head>\n" + _RAF_FALLBACK, 1)
    return patched


class EpisodeRecorder:
    """Record one episode's Meshcat animation and save a standalone HTML."""

    def __init__(self, rig, fps: float = RECORDING_FPS) -> None:
        self.rig = rig
        self.fps = float(fps)
        self.meshcat = rig.station.meshcat
        self._active = False

    def start(self) -> None:
        simulator_time = self.rig.station.simulator.get_context().get_time()
        self.meshcat.StartRecording(frames_per_second=self.fps)
        recording = self.meshcat.get_mutable_recording()
        # Sim time accumulates across episodes on a persistent rig, and the
        # animation maps time to frame index; without this offset the replay
        # would begin with a huge empty prefix.
        recording.set_start_time(simulator_time)
        # Re-publish everything: SetObject calls are never animated, so the
        # static scene must be re-emitted after StartRecording (see module
        # docstring). The goal markers take time_in_recording so their
        # transforms land inside the animation.
        self.rig.station.diagram.ForcedPublish(self.rig.station.context)
        self.rig.station.publish_goal(simulator_time)
        self._active = True

    def save(self, path: str | Path) -> Path:
        if not self._active:
            raise RuntimeError("start() was never called")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.meshcat.StopRecording()
        # Contact arrows are a live debugging aid only. Drake's
        # ContactVisualizer resurrects transient contact pairs with SetObject,
        # which a MeshcatAnimation never records; a saved replay would show
        # the persistent table-T arrows but lose the pusher-T force the
        # moment contact first breaks. Hide the whole subtree before the
        # snapshot rather than ship a partial, misleading subset; restore
        # afterwards so the live view of the next episode keeps its arrows.
        self.meshcat.SetProperty("/drake/contact_forces", "visible", False)
        self.meshcat.PublishRecording()
        path.write_text(make_portable(self.meshcat.StaticHtml()))
        self.meshcat.SetProperty("/drake/contact_forces", "visible", True)
        self.meshcat.DeleteRecording()
        self._active = False
        return path
