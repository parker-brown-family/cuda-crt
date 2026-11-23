#!/usr/bin/env python3
"""Local CRT controls. Stdlib only; run --help for paths and port.

GET /api/dials, /api/presets, /api/knobs; POST /api/knobs with
{"values": {"dial": number, ...}, "preset": name_or_null} (partial sets allowed).
Every successful write publishes ALL values by same-directory os.replace.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<meta name="fancy-pants-read-only" content="Interactive CRT renderer control panel, in the product's own design">
<title>CRT controls</title>
<style>
*{box-sizing:border-box}body{margin:0;color:#dcf5e7;background:#09120e;font:15px/1.45 ui-monospace,monospace}
body:before{content:"";position:fixed;inset:0;pointer-events:none;background:repeating-linear-gradient(transparent 0 3px,#0002 3px 4px)}
main{max-width:600px;padding:20px 16px 48px;margin:auto}h1{font-size:25px;letter-spacing:.06em;margin:0}
h2{font-size:17px;color:#a9f3c6;margin:0 0 15px}p{margin:6px 0 12px;color:#a9c2b2}
header{position:sticky;top:0;z-index:1;background:#09120ef5;padding-bottom:12px;border-bottom:1px solid #345744}
#status{font-size:12px;min-height:18px;color:#a9f3c6;overflow-wrap:anywhere}#status.error{color:#ffb3a1}
section{padding:18px 0;border-bottom:1px solid #294536}.buttons{display:flex;flex-wrap:wrap;gap:8px}
button{font:inherit;background:linear-gradient(#264a35,#162c20);border:1px solid #4a7e5e;border-radius:7px;color:#e1f9e9;padding:8px 11px;cursor:pointer;overflow-wrap:anywhere;max-width:100%}
button:hover,button[aria-pressed="true"]{background:#376546}button:disabled{opacity:.4;cursor:default}
button:focus-visible,input:focus-visible{outline:2px solid #b9ffcf;outline-offset:4px}
.dial{margin:0 0 21px}.dial:last-child{margin-bottom:0}.top{display:flex;align-items:baseline;justify-content:space-between;gap:10px}
label{overflow-wrap:anywhere;min-width:0}output{color:#b7ffd1;white-space:nowrap;font-variant-numeric:tabular-nums}
input[type=range]{display:block;width:100%;margin:12px 0;accent-color:#94efaf;min-height:24px;cursor:pointer}
.help{font-size:12px;color:#adc2b4;margin:0;overflow-wrap:anywhere}.caption{font-size:12px}noscript{color:#ffb3a1}
@media(max-width:360px){main{padding:14px 12px}.top{flex-wrap:wrap}h1{font-size:22px}}
</style></head><body><main>
<header><h1>CRT controls</h1><p>Shape the phosphor, glass and signal.</p><div id="status" role="status" aria-live="polite">Connecting…</div></header>
<noscript>Enable JavaScript to use the sliders.</noscript>
<section aria-label="Presets"><h2>Presets</h2><div id="presets" class="buttons"></div></section>
<section aria-label="Ghost shortcuts"><h2>Ghost linger</h2><div id="ghost" class="buttons"></div><p class="caption">None turns off the ghost. Short uses defaults; Long uses the slider limits.</p></section>
<div id="groups"></div>
<section><button id="reset" type="button" disabled>Reset all dials</button></section>
</main><script>
'use strict';
let dials=[], presets={}, values=Object.create(null), preset=null, timer=null, busy=false, dirty=false;
let generation=0;
const controls=new Map(), presetButtons=new Map();
const status=document.getElementById('status');
function message(text,error=false){status.textContent=text;status.className=error?'error':'';}
async function api(path,body){
  const options=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)};
  const response=await fetch(path,options);
  const data=await response.json();
  if(!response.ok)throw new Error(data.error||'Request failed');
  return data;
}
function paint(){
  for(const d of dials){
    const c=controls.get(d.name);c.input.value=values[d.name];
    const [base,exponent='0']=String(d.step).toLowerCase().split('e');
    const digits=Math.min(12,Math.max(0,(base.split('.')[1]||'').length-Number(exponent)));
    c.output.textContent=Number(values[d.name].toFixed(digits))+(d.unit?' '+d.unit:'');
    c.input.setAttribute('aria-valuetext',c.output.textContent);
  }
  for(const [name,button] of presetButtons)button.setAttribute('aria-pressed',String(name===preset));
}
function changed(){
  generation++;dirty=true;paint();message('Saving…');
  // A fixed 35 ms coalescing window: sustained input cannot postpone writes.
  if(timer===null&&!busy)timer=setTimeout(flush,35);
}
async function flush(){
  timer=null;if(busy||!dirty)return;
  busy=true;dirty=false;const sent=generation;
  try{
    const saved=await api('/api/knobs',{values:{...values},preset});
    if(sent===generation){values=Object.assign(Object.create(null),saved.values);preset=saved.preset;paint();message('Saved · revision '+saved.rev);}
  }catch(error){message('Not saved: '+error.message+' · move a slider to retry',true);}
  finally{busy=false;if(dirty&&timer===null)timer=setTimeout(flush,0);}
}
function button(parent,text,action){
  const b=document.createElement('button');b.type='button';b.textContent=text;b.addEventListener('click',action);parent.append(b);return b;
}
async function start(){
  [dials,presets]=await Promise.all([api('/api/dials'),api('/api/presets')]);
  const saved=await api('/api/knobs');values=Object.assign(Object.create(null),saved.values);preset=saved.preset;
  const groups=new Map();
  for(const [i,d] of dials.entries()){
    if(!groups.has(d.group)){
      const section=document.createElement('section'),title=document.createElement('h2');title.textContent=d.group;
      section.append(title);document.getElementById('groups').append(section);groups.set(d.group,section);
    }
    const row=document.createElement('div'),top=document.createElement('div'),label=document.createElement('label');
    row.className='dial';top.className='top';label.textContent=d.label;label.htmlFor='dial-'+i;
    const output=document.createElement('output'),input=document.createElement('input'),help=document.createElement('p');
    input.id='dial-'+i;output.setAttribute('for',input.id);input.type='range';input.min=d.min;input.max=d.max;input.step=d.step;
    help.id='help-'+i;help.className='help';help.textContent=d.help;input.setAttribute('aria-describedby',help.id);
    input.addEventListener('input',()=>{values[d.name]=Number(input.value);preset=null;changed();});
    top.append(label,output);row.append(top,input,help);groups.get(d.group).append(row);controls.set(d.name,{input,output});
  }
  const list=document.getElementById('presets');
  for(const [name,settings] of Object.entries(presets)){
    presetButtons.set(name,button(list,name,()=>{Object.assign(values,settings);preset=name;changed();}));
  }
  if(!Object.keys(presets).length){const p=document.createElement('p');p.textContent='No presets defined.';list.append(p);}
  for(const mode of ['None','Short','Long']){
    const b=button(document.getElementById('ghost'),mode,()=>{
      for(const name of ['ghost','ghost_decay']){
        const d=dials.find(item=>item.name===name);if(!d)continue;
        values[name]=mode==='None'?(name==='ghost'?0:d.min):mode==='Short'?d.default:d.max;
      }
      preset=null;changed();
    });b.disabled=!dials.some(d=>d.name==='ghost'||d.name==='ghost_decay');
  }
  const reset=document.getElementById('reset');reset.disabled=false;
  reset.addEventListener('click',()=>{for(const d of dials)values[d.name]=d.default;preset=null;changed();});
  paint();message('Ready · revision '+saved.rev);
  // Another writer (DOOM's own menu) may move a dial while this page is open.
  let rev=saved.rev;
  setInterval(async()=>{
    if(document.hidden||busy||dirty||timer!==null)return;
    const at=generation;
    try{
      const now=await api('/api/knobs');
      if(at!==generation||busy||dirty||now.rev===rev)return;
      rev=now.rev;values=Object.assign(Object.create(null),now.values);preset=now.preset;paint();
    }catch(error){}
  },1000);
}
start().catch(error=>message('Could not load controls: '+error.message,true));
</script></body></html>'''


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("dial values must be numbers")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("dial values must be finite")
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Knobs:
    """One writer owns the file; the lock serializes concurrent HTTP updates."""

    def __init__(self, dials, presets, path):
        if not isinstance(dials, list):
            raise ValueError("dials must be a list")
        self.dials = copy.deepcopy(dials)
        self.by_name = {}
        for dial in self.dials:
            if not isinstance(dial, dict):
                raise ValueError("each dial must be an object")
            for key in ("name", "label", "group", "unit", "help"):
                if not isinstance(dial.get(key), str):
                    raise ValueError("each dial needs string " + key)
            name = dial["name"]
            if not name or name in self.by_name:
                raise ValueError("dial names must be nonempty and unique")
            lo, hi = number(dial.get("min")), number(dial.get("max"))
            if lo > hi or number(dial.get("step")) <= 0:
                raise ValueError("dial limits or step are invalid")
            if not lo <= number(dial.get("default")) <= hi:
                raise ValueError("default is outside dial limits: " + name)
            self.by_name[name] = dial
        if not isinstance(presets, dict):
            raise ValueError("presets must be an object")
        self.presets = {}
        for name, values in presets.items():
            if not isinstance(name, str) or not name:
                raise ValueError("preset names must be nonempty strings")
            self.presets[name] = self.clamp(values)
        self.path = Path(path)
        self.lock = threading.Lock()
        self.current = {"rev": 0, "preset": None,
                        "values": {d["name"]: d["default"] for d in self.dials}}
        if self.path.exists():
            old = read_json(self.path)
            if not isinstance(old, dict) or type(old.get("rev")) is not int or old["rev"] < 0:
                raise ValueError("existing knobs file has an invalid revision")
            if not isinstance(old.get("values"), dict):
                raise ValueError("existing knobs file has invalid values")
            # New metadata can retire dials; only restore values still declared.
            restored = {k: v for k, v in old["values"].items() if k in self.by_name}
            self.current["values"].update(self.clamp(restored))
            self.current["rev"] = old["rev"]
            if isinstance(old.get("preset"), str) and old["preset"] in self.presets:
                self.current["preset"] = old["preset"]
        self.set({}, self.current["preset"])  # Publish defaults/restored values before accepting clients.

    def clamp(self, values):
        if not isinstance(values, dict):
            raise ValueError("values must be an object")
        result = {}
        for name, value in values.items():
            if name not in self.by_name:
                raise ValueError("unknown dial: " + name)
            dial = self.by_name[name]
            result[name] = min(dial["max"], max(dial["min"], number(value)))
        return result

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.current)

    def set(self, values, preset=None):
        values = self.clamp(values)
        if preset is not None and (not isinstance(preset, str) or preset not in self.presets):
            raise ValueError("unknown preset")
        with self.lock:
            next_state = {"rev": self.current["rev"] + 1, "preset": preset,
                          "values": {**self.current["values"], **values}}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                 prefix="." + self.path.name + ".", delete=False) as f:
                    temporary = f.name
                    json.dump(next_state, f, allow_nan=False, separators=(",", ":"))
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temporary, self.path)
            finally:
                if temporary is not None and os.path.exists(temporary):
                    os.unlink(temporary)
            self.current = next_state
            return copy.deepcopy(next_state)


class Handler(BaseHTTPRequestHandler):
    def parse_request(self):
        if not super().parse_request():
            return False
        return self.host_ok()

    def log_message(self, *_args):
        pass

    def reply(self, status, body, content_type="application/json"):
        data = (body if isinstance(body, str) else json.dumps(body, allow_nan=False)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; "
                         "style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; "
                         "frame-ancestors 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(data)

    def host_ok(self):
        hosts = self.headers.get_all("Host", [])
        port = self.server.server_port
        if len(hosts) != 1 or hosts[0] not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            self.reply(403, {"error": "Host refused"})
            return False
        return True

    def do_GET(self):
        state = self.server.knobs
        if self.path == "/":
            self.reply(200, PAGE, "text/html")
        elif self.path == "/api/dials":
            self.reply(200, state.dials)
        elif self.path == "/api/presets":
            self.reply(200, state.presets)
        elif self.path == "/api/knobs":
            self.reply(200, state.snapshot())
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        origins = self.headers.get_all("Origin", [])
        if origins and origins != ["http://" + self.headers["Host"]]:
            self.reply(403, {"error": "Origin refused"})
            return
        content_types = self.headers.get_all("Content-Type", [])
        if len(content_types) != 1 or self.headers.get_content_type() != "application/json":
            self.reply(415, {"error": "application/json required"})
            return
        if self.path != "/api/knobs":
            self.reply(404, {"error": "not found"})
            return
        try:
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get("Transfer-Encoding") or len(lengths) != 1:
                raise ValueError("one Content-Length required; no Transfer-Encoding")
            length = int(lengths[0])
            if not 0 < length <= 1024 * 1024:
                raise ValueError("body must be between 1 byte and 1 MiB")
            self.connection.settimeout(3)
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("incomplete body")
            request = json.loads(body)
            if not isinstance(request, dict) or "values" not in request or set(request) - {"values", "preset"}:
                raise ValueError("expected values and optional preset")
            result = self.server.knobs.set(request["values"], request.get("preset"))
        except (ValueError, UnicodeError, TimeoutError, RecursionError) as error:
            self.reply(400, {"error": str(error)})
            return
        except OSError:
            self.reply(500, {"error": "could not publish knobs file"})
            return
        self.reply(200, result)


def make_server(dials, presets, path, port=8787):
    # No caller-supplied bind address: this API cannot expose the panel remotely.
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    try:
        server.knobs = Knobs(read_json(dials), read_json(presets) if presets else {}, path)
    except Exception:
        server.server_close()
        raise
    return server


def main():
    tools = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--dials", type=Path)
    parser.add_argument("--presets", type=Path)
    parser.add_argument("--file", type=Path, default=tools.parent / "out/knobs.json")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535 (0 selects a free port)")
    dials = args.dials or (tools / "dials.json" if (tools / "dials.json").exists() else tools / "samples/dials.json")
    presets = args.presets or (tools / "presets.json" if (tools / "presets.json").exists()
                               else tools / "samples/presets.json" if dials == tools / "samples/dials.json" else None)
    try:
        server = make_server(dials, presets, args.file, args.port)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"CRT controls: http://127.0.0.1:{server.server_port}/", flush=True)
    print(f"Dials: {dials}; presets: {presets or '(none)'}; knobs: {args.file}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
