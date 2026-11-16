import QtQuick
import Quickshell
import Quickshell.Wayland

// Stands in for Omarchy's dock: a TOP layer along the bottom of the monitor whose
// whole strip takes the pointer. Over a fullscreen window, Hyprland ignores it.
ShellRoot {
  PanelWindow {
    anchors { bottom: true; left: true; right: true }
    implicitHeight: 207
    exclusionMode: ExclusionMode.Ignore
    WlrLayershell.layer: WlrLayer.Top
    WlrLayershell.namespace: "test-dock"
    color: "#30406080"
  }
}
