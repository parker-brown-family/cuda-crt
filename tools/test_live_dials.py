#!/usr/bin/env python3
"""The panel and DOOM's menu share one look: python3 tools/test_live_dials.py."""
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import live_dials

loader = SourceFileLoader("crt_config", str(TOOLS / "crt-config"))
crt_config = module_from_spec(spec_from_loader("crt_config", loader))
loader.exec_module(crt_config)


class SharedLook(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.state = Path(temp.name)

    def game_writes(self, text):
        for name in ("tube-profile.set", "set"):
            (self.state / name).write_text(text)

    def test_opening_writes_nothing_live(self):
        (self.state / "set").write_text("glow=0.15\n")
        crt_config.LiveKnobs(self.state)
        self.assertEqual((self.state / "set").read_text(), "glow=0.15\n")
        self.assertFalse((self.state / "tube-profile.set").exists())

    def test_panel_shows_then_keeps_what_the_game_set(self):
        knobs = crt_config.LiveKnobs(self.state)
        rev = knobs.snapshot()["rev"]
        self.game_writes("scan=0.4,other=7\n")
        self.assertEqual(knobs.snapshot()["values"]["scan"], 0.4)
        self.assertGreater(knobs.snapshot()["rev"], rev)
        knobs.set({"glow": 0.5})
        text = (self.state / "set").read_text()
        for pair in ("scan=0.4", "glow=0.5", "other=7"):
            self.assertIn(pair, text)
        self.assertEqual(text, (self.state / "tube-profile.set").read_text())

    def test_a_slider_move_without_a_read_first_does_not_undo_the_game(self):
        knobs = crt_config.LiveKnobs(self.state)
        self.game_writes("curv=2\n")
        self.assertEqual(knobs.set({"glow": 0.5})["values"]["curv"], 2)

    def test_the_saved_look_outlives_the_session(self):
        knobs = crt_config.LiveKnobs(self.state)
        self.game_writes("curv=2\n")
        knobs.snapshot()
        (self.state / "set").write_text("curv=1\n")  # the launcher puts the old set back on exit
        self.assertEqual(knobs.snapshot()["values"]["curv"], 2)

    def test_a_bad_number_is_skipped(self):
        knobs = crt_config.LiveKnobs(self.state)
        self.game_writes("curv=2,scan=nonsense,glow=nan\n")
        values = knobs.snapshot()["values"]
        self.assertEqual((values["curv"], values["glow"]), (2, 0.4))

    def test_every_dial_the_menu_offers_is_one_the_pass_reads(self):
        source = (TOOLS.parent / "hypr/cuda-pass.cpp").read_text()
        for dial in live_dials.load()[0]:
            self.assertIn('"%s"' % dial["name"], source)


if __name__ == "__main__":
    unittest.main()
