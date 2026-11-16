#!/usr/bin/env python3
"""CPU-only HTTP/file integration checks: python3 tools/test_knobs.py."""

import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

import knobs


SAMPLES = Path(__file__).resolve().parent / "samples"


class KnobsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "out/knobs.json"
        self.server = knobs.make_server(SAMPLES / "dials.json", SAMPLES / "presets.json", self.path, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.port = self.server.server_port
        self.host = f"127.0.0.1:{self.port}"
        self.names = {d["name"] for d in knobs.read_json(SAMPLES / "dials.json")}

    def stop_server(self):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()

    def request(self, method="GET", path="/api/knobs", payload=None, headers=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        self.addCleanup(conn.close)
        request_headers = {"Host": self.host}
        if payload is not None:
            raw = json.dumps(payload)
            request_headers["Content-Type"] = "application/json"
        request_headers.update(headers or {})
        conn.request(method, path, body=raw, headers=request_headers)
        response = conn.getresponse()
        body = response.read()
        content_type = response.getheader("Content-Type")
        result = json.loads(body) if content_type.startswith("application/json") else body.decode()
        status = response.status
        conn.close()
        return status, result

    def test_page_and_metadata(self):
        status, page = self.request(path="/")
        self.assertEqual(status, 200)
        self.assertIn('name="viewport"', page)
        self.assertIn("setTimeout(flush,35)", page)
        self.assertNotIn("innerHTML", page)
        self.assertNotIn('src="http', page)
        self.assertIn("role=\"status\"", page)
        self.assertEqual(self.request(path="/api/dials"), (200, knobs.read_json(SAMPLES / "dials.json")))
        self.assertEqual(self.request(path="/api/presets"), (200, knobs.read_json(SAMPLES / "presets.json")))
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        self.assertEqual(self.request(path="/does-not-exist")[0], 404)

    def test_valid_write_all_dials_and_revision(self):
        initial = self.request()[1]
        self.assertIsNone(initial["preset"])
        for value in (100, 140):
            status, saved = self.request("POST", payload={"values": {"decay_g": value}, "preset": "P39 long green"},
                                         headers={"Origin": "http://" + self.host})
            self.assertEqual(status, 200)
            self.assertEqual(saved["values"]["decay_g"], value)
            self.assertEqual(saved["rev"], initial["rev"] + 1)
            self.assertEqual(set(saved["values"]), self.names)
            self.assertEqual(saved["preset"], "P39 long green")
            self.assertEqual(knobs.read_json(self.path), saved)
            initial = saved
        # Manual updates clear the preset; unchanged values stay present.
        status, saved = self.request("POST", payload={"values": {"curv": 0.5}})
        self.assertEqual(status, 200)
        self.assertIsNone(saved["preset"])
        self.assertEqual(saved["values"]["decay_g"], 140)

    def test_clamp_and_reject_invalid_values_without_writing(self):
        status, saved = self.request("POST", payload={"values": {"curv": 99, "ghost": -1}})
        self.assertEqual(status, 200)
        self.assertEqual(saved["values"]["curv"], 3)
        self.assertEqual(saved["values"]["ghost"], 0)
        before = self.path.read_bytes()
        for values in ({"not-a-dial": 1}, {"curv": True}, {"curv": None}, {"curv": "1"},
                       {"curv": float("nan")}, {"curv": float("inf")}, {"curv": 10**400}, [], {"curv": 1, "oops": 3}):
            with self.subTest(values=values):
                self.assertEqual(self.request("POST", payload={"values": values})[0], 400)
                self.assertEqual(self.path.read_bytes(), before)
        for payload in ({"values": {}, "preset": "missing"}, {"values": {}, "extra": 0}, [], {}):
            self.assertEqual(self.request("POST", payload=payload)[0], 400)
            self.assertEqual(self.path.read_bytes(), before)

    def test_host_guard(self):
        before = self.path.read_bytes()
        for host in ("evil.example", f"evil.example:{self.port}", f"127.0.0.1:{self.port}.evil", "localhost"):
            with self.subTest(host=host):
                self.assertEqual(self.request(path="/", headers={"Host": host})[0], 403)
                self.assertEqual(self.request("POST", payload={"values": {}}, headers={"Host": host})[0], 403)
        self.assertEqual(self.request("OPTIONS", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.request(headers={"Host": f"localhost:{self.port}"})[0], 200)
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        self.addCleanup(conn.close)
        conn.putrequest("GET", "/", skip_host=True)
        conn.putheader("Host", self.host)
        conn.putheader("Host", self.host)
        conn.endheaders()
        self.assertEqual(conn.getresponse().status, 403)
        self.assertEqual(self.path.read_bytes(), before)

    def test_origin_and_content_type_guards(self):
        before = self.path.read_bytes()
        for origin in ("https://evil.example", "null", "http://" + self.host + ".evil", "https://" + self.host):
            with self.subTest(origin=origin):
                self.assertEqual(self.request("POST", payload={"values": {}}, headers={"Origin": origin})[0], 403)
        for content_type in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
            self.assertEqual(self.request("POST", payload={"values": {}}, headers={"Content-Type": content_type})[0], 415)
        self.assertEqual(self.request("POST", raw='{"values":{}}')[0], 415)
        self.assertEqual(self.request("POST", raw="not json", headers={"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.request("POST", payload={"values": {}},
                                     headers={"Host": f"localhost:{self.port}", "Origin": f"http://localhost:{self.port}"})[0], 200)

    def test_200_atomic_writes_with_concurrent_reader(self):
        stop = threading.Event()
        ready = threading.Event()
        errors, observations = [], []
        initial_rev = knobs.read_json(self.path)["rev"]

        def reader():
            last_rev = initial_rev
            while not stop.is_set():
                try:
                    state = knobs.read_json(self.path)
                    if set(state) != {"rev", "preset", "values"} or set(state["values"]) != self.names:
                        raise AssertionError("incomplete state observed")
                    if state["rev"] < last_rev:
                        raise AssertionError("revision went backwards")
                    last_rev = state["rev"]
                    observations.append(last_rev)
                    ready.set()
                except Exception as error:
                    errors.append(repr(error))
                    ready.set()
                    break

        worker = threading.Thread(target=reader)
        worker.start()
        try:
            self.assertTrue(ready.wait(2))
            for i in range(200):
                status, saved = self.request("POST", payload={"values": {"decay_g": 5 + i}})
                self.assertEqual(status, 200)
                self.assertEqual(saved["rev"], initial_rev + i + 1)
        finally:
            stop.set()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors, errors)
        self.assertGreater(len(observations), 200)
        self.assertGreater(len(set(observations)), 1)
        self.assertEqual(knobs.read_json(self.path)["rev"], initial_rev + 200)
        self.assertEqual(list(self.path.parent.glob(".knobs.json.*")), [])

    def test_concurrent_clients_do_not_lose_updates(self):
        start = self.request()[1]["rev"]
        errors, revisions = [], []

        def post(name, value):
            try:
                status, saved = self.request("POST", payload={"values": {name: value}})
                if status != 200:
                    raise AssertionError(saved)
                revisions.append(saved["rev"])
            except Exception as error:
                errors.append(repr(error))

        workers = [threading.Thread(target=post, args=("curv", 2)),
                   threading.Thread(target=post, args=("decay_g", 123))]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5)
        self.assertFalse(errors, errors)
        self.assertEqual(sorted(revisions), [start + 1, start + 2])
        current = self.request()[1]
        self.assertEqual(current["values"]["curv"], 2)
        self.assertEqual(current["values"]["decay_g"], 123)

    def test_restart_restores_values_and_increments_revision(self):
        saved = self.request("POST", payload={"values": {"curv": 1.5}})[1]
        restored = knobs.Knobs(knobs.read_json(SAMPLES / "dials.json"), knobs.read_json(SAMPLES / "presets.json"), self.path)
        self.assertEqual(restored.snapshot()["rev"], saved["rev"] + 1)
        self.assertEqual(restored.snapshot()["values"], saved["values"])

    def test_occupied_port_does_not_write_file(self):
        before = self.path.read_bytes()
        with self.assertRaises(OSError):
            knobs.make_server(SAMPLES / "dials.json", SAMPLES / "presets.json", self.path, self.port)
        self.assertEqual(self.path.read_bytes(), before)

    def test_arbitrary_names_groups_and_count(self):
        dials = [{"name": f"d{i}", "label": f"Control {i}", "group": f"Unfamiliar group {i % 7}",
                  "min": -10, "max": 10, "step": 0.01, "default": 0, "unit": "u", "help": "Help"} for i in range(90)]
        dials.append({**dials[0], "name": "__proto__", "group": "<script>not HTML</script>"})
        state = knobs.Knobs(dials, {}, Path(self.temp.name) / "arbitrary.json")
        self.assertEqual(len(state.set({"d89": 30})["values"]), 91)
        self.assertEqual(state.snapshot()["values"]["d89"], 10)
        self.assertIn("__proto__", state.snapshot()["values"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
