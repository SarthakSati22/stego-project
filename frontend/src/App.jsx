import { useEffect, useState } from "react";
import "./App.css";

const API = import.meta.env.VITE_API ?? "http://127.0.0.1:8000";

async function api(path, body) {
  const res = await fetch(API + path, body ? {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  } : undefined);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    throw new Error(typeof d === "string" ? d : JSON.stringify(d) || res.statusText);
  }
  return data;
}

const fmt = (x, d = 1) => (x === null || x === undefined ? "-" : Number(x).toFixed(d));

function Score({ v }) {
  return (
    <>
      <div className="meter"><i style={{ width: `${(v * 100).toFixed(1)}%` }} /></div>
      <span className="hint">{v.toFixed(4)}</span>{" "}
      <span className={"flag " + (v > 0.5 ? "yes" : "no")}>{v > 0.5 ? "flagged" : "clean"}</span>
    </>
  );
}

export default function App() {
  const [opts, setOpts] = useState({ strategies: [], channels: [] });
  const [clips, setClips] = useState([]);
  const [clip, setClip] = useState("");
  const [strategy, setStrategy] = useState("");
  const [secret, setSecret] = useState("hello stego 123");
  const [pass, setPass] = useState("pw");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [res, setRes] = useState(null);

  const [channel, setChannel] = useState("");
  const [pass2, setPass2] = useState("pw");
  const [exBusy, setExBusy] = useState(false);
  const [exErr, setExErr] = useState("");
  const [out, setOut] = useState(null);

  useEffect(() => {
    Promise.all([api("/options"), api("/clips?split=test&limit=40")])
      .then(([o, c]) => {
        setOpts(o);
        setClips(c);
        setClip(c[0]?.clip_id ?? "");
        setStrategy(o.strategies.includes("fixed_50_50") ? "fixed_50_50" : o.strategies[0] ?? "");
        setChannel(o.channels.includes("mp3_128+h264_crf23") ? "mp3_128+h264_crf23" : o.channels[0] ?? "");
      })
      .catch((e) => setError("Cannot reach the backend: " + e.message));
  }, []);

  async function hide() {
    setError("");
    setBusy(true);
    try {
      const r = await api("/embed", { clip_id: clip, secret, passphrase: pass, strategy });
      setRes(r);
      setPass2(pass);
      setOut(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function extract() {
    setExErr("");
    setOut(null);
    setExBusy(true);
    try {
      setOut(await api("/extract", { job_id: res.job_id, passphrase: pass2, channel }));
    } catch (e) {
      setExErr(e.message);
    } finally {
      setExBusy(false);
    }
  }

  const c = res?.detection_clean;
  const s = res?.detection_stego;

  return (
    <main>
      <h1>Cross-Modal Stego Lab</h1>
      <p className="sub">
        Hide an encrypted secret across the audio and video of a clip, check how detectable it
        is, then send it through a compression channel and read it back.
      </p>

      <section className="card">
        <h2>1. Hide a secret</h2>
        <div className="grid">
          <div>
            <label htmlFor="clip">Clip</label>
            <select id="clip" value={clip} onChange={(e) => setClip(e.target.value)}>
              {clips.map((x) => (
                <option key={x.clip_id} value={x.clip_id}>{x.clip_id} ({x.emotion})</option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="strategy">Strategy</label>
            <select id="strategy" value={strategy} onChange={(e) => setStrategy(e.target.value)}>
              {opts.strategies.map((x) => <option key={x}>{x}</option>)}
            </select>
          </div>
          <div>
            <label htmlFor="secret">Secret (max 128 characters)</label>
            <input id="secret" maxLength={128} value={secret} onChange={(e) => setSecret(e.target.value)} />
          </div>
          <div>
            <label htmlFor="pass">Passphrase</label>
            <input id="pass" value={pass} onChange={(e) => setPass(e.target.value)} />
          </div>
        </div>
        <div className="row">
          <button onClick={hide} disabled={busy || !clip}>Hide secret</button>
          <span className="hint">{busy ? "Working..." : "Takes a few seconds."}</span>
        </div>
        {error && <div className="err">{error}</div>}
      </section>

      {res && (
        <section className="card">
          <h2>Result</h2>
          <div className="hint">Job <code>{res.job_id}</code></div>
          <div className="legend" style={{ marginTop: 10 }}>How the {res.bits} bits were split</div>
          <div className="split">
            <div className="a" style={{ width: `${(100 * res.bits_audio) / res.bits}%` }} />
            <div className="v" style={{ width: `${(100 * res.bits_video) / res.bits}%` }} />
          </div>
          <div className="legend">
            <span style={{ color: "#3b82f6" }}>&#9632;</span> audio <b>{res.bits_audio}</b>
            &nbsp; <span style={{ color: "#f59e0b" }}>&#9632;</span> video <b>{res.bits_video}</b>
          </div>

          <div className="metrics">
            <div className="metric"><b>{fmt(res.quality.snr_db)}</b><span>Audio SNR (dB, higher = less damage)</span></div>
            <div className="metric"><b>{fmt(res.quality.psnr_db)}</b><span>Video PSNR (dB, higher = less damage)</span></div>
            <div className="metric"><b>{fmt(res.quality.ssim, 3)}</b><span>Video SSIM (1.0 = identical)</span></div>
          </div>

          <h2 style={{ marginTop: 18 }}>Can a detector find it?</h2>
          <div className="hint">
            Score from 0 (looks clean) to 1 (looks like it hides something). Above 0.5 counts as flagged.
          </div>
          {c && s ? (
            <table>
              <thead><tr><th>Detector</th><th>Original clip</th><th>Clip with secret</th></tr></thead>
              <tbody>
                {[["Audio", "audio"], ["Video", "video"], ["Fused (final)", "fused"]].map(([name, k]) => (
                  <tr key={k}><td>{name}</td><td><Score v={c[k]} /></td><td><Score v={s[k]} /></td></tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="hint">Detection scores are not available (detectors did not load).</div>
          )}
          <div className="row">
            <a href={`${API}/download/${res.job_id}/audio`}>Download stego audio</a>
            <a href={`${API}/download/${res.job_id}/video`}>Download stego video</a>
          </div>
        </section>
      )}

      {res && (
        <section className="card">
          <h2>2. Send it through a channel and read it back</h2>
          <div className="grid">
            <div>
              <label htmlFor="channel">Channel (real compression)</label>
              <select id="channel" value={channel} onChange={(e) => setChannel(e.target.value)}>
                {opts.channels.map((x) => <option key={x}>{x}</option>)}
              </select>
            </div>
            <div>
              <label htmlFor="pass2">Passphrase</label>
              <input id="pass2" value={pass2} onChange={(e) => setPass2(e.target.value)} />
            </div>
          </div>
          <div className="row">
            <button onClick={extract} disabled={exBusy}>Extract secret</button>
            <span className="hint">{exBusy ? "Working..." : ""}</span>
          </div>
          {exErr && <div className="err">{exErr}</div>}
          {out && (
            <div className="row">
              <span className={"badge " + (out.recovered ? "ok" : "fail")}>
                {out.recovered ? "Recovered" : "Not recovered"}
              </span>
              <span>Secret: <code>{out.recovered ? out.secret : "-"}</code></span>
            </div>
          )}
        </section>
      )}
    </main>
  );
}