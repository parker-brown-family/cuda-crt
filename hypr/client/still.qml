import QtQuick
import Quickshell

// The same stand-in terminal with its text complete and nothing moving: a still
// window, for the idle and failure checks.
ShellRoot {
  FloatingWindow {
    id: win
    title: "still"
    implicitWidth: 1500
    implicitHeight: 800
    color: "#080c09"
    property var lines: ["$ cargo build --release", "   Compiling phosphor-surface v0.1.0", "   Compiling crt-glass v0.2.0",
      "    Finished release [optimized] target(s) in 4.21s", "$ ./bin/stage1_test", "all checks passed",
      "$ nvidia-smi --query-gpu=name --format=csv,noheader", "NVIDIA GeForce RTX 3080 Laptop GPU", "$ git log --oneline -3",
      "bcfd334 Drops land where the screen shows them", "$ clear"]
    property string all: lines.join("\n")
    property int shown: 100000
    Text {
      x: 28; y: 22
      text: win.all.substring(0, win.shown) + "█"
      color: "#46ff82"
      font.family: "JetBrainsMono Nerd Font"
      font.pixelSize: 30
      lineHeight: 1.25
    }
  }
}
