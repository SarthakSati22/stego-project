import { useEffect, useState } from "react";

const API = import.meta.env.VITE_API || (import.meta.env.PROD ? "" : "http://127.0.0.1:8000");

async function call(path, body) {
  const r = await fetch(API + path, body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : undefined);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? data));
  }
  return data;
}

const fmt = (x, d = 1) => (x == null ? "-" : Number(x).toFixed(d));

function StatusChip({ name, state }) {
  const isOk = state === "ok";
  return (
    <span
      className={`status-chip ${isOk ? "clean" : "flagged"}`}
      style={{ marginRight: 6, marginBottom: 4 }}
    >
      {name}: {isOk ? "Surviving" : "Destroyed"}
    </span>
  );
}

export default function ThreeCarrier() {
  const [clips, setClips] = useState([]);
  const [opts, setOpts] = useState(null);
  const [clip, setClip] = useState("");
  const [secret, setSecret] = useState("Cross-modal distributed secret 2026");
  const [pass, setPass] = useState("demo-tri-pass");
  const [plan, setPlan] = useState("erasure_2of3");
  const [scenario, setScenario] = useState("kill_video");
  const [job, setJob] = useState(null);
  const [res, setRes] = useState(null);
  const [table, setTable] = useState(null);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");

  useEffect(() => {
    Promise.all([call("/clips?split=test&limit=20"), call("/options3")])
      .then(([c, o]) => {
        setClips(c);
        setOpts(o);
        if (c.length) setClip(c[0].clip_id);
      })
      .catch((e) => {
        setErr("Cannot reach 3-carrier API. Launch with: python -m uvicorn api.app3:app --port 8000 (" + e.message + ")");
      });
  }, []);

  const run = async (label, fn) => {
    setErr("");
    setBusy(label);
    try {
      await fn();
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy("");
    }
  };

  const hide = () => run("Splitting payload and encoding Reed-Solomon parity...", async () => {
    setRes(null);
    setTable(null);
    setJob(await call("/embed3", { clip_id: clip, secret, passphrase: pass, plan }));
  });

  const attack = () => run("Applying carrier destruction scenario...", async () => {
    setTable(null);
    setRes(await call("/extract3", { job_id: job.job_id, passphrase: pass, scenario }));
  });

  const sweep = () => run("Executing full scenario resilience sweep...", async () => {
    const rows = [];
    for (const s of opts.scenarios) {
      rows.push(await call("/extract3", { job_id: job.job_id, passphrase: pass, scenario: s }));
      setTable([...rows]);
    }
  });

  const planInfo = opts?.plans.find((p) => p.name === plan);

  return (
    <div>
      <section className="panel-card">
        <div className="panel-header">
          <div>
            <div className="panel-title">
              <span className="step-tag">Tri-Modal Parity</span>
              <span>Audio + Video + Image Distributed Steganography</span>
            </div>
            <p className="panel-desc">
              The secret payload is partitioned using $k$-of-$n$ erasure coding across three distinct modalities.
              Even if a channel attacker completely corrupts or strips an entire carrier, the secret remains 100% reconstructible.
            </p>
          </div>
        </div>

        <div className="form-grid">
          <div className="form-group">
            <label className="form-label">Carrier Video/Audio Clip</label>
            <select value={clip} onChange={(e) => setClip(e.target.value)}>
              {clips.map((c) => (
                <option key={c.clip_id} value={c.clip_id}>
                  {c.clip_id}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Erasure Coding Strategy</label>
            <select value={plan} onChange={(e) => setPlan(e.target.value)}>
              {opts?.plans.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name.replace(/_/g, " ").toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group full-width">
            <label className="form-label">
              <span>Secret Message</span>
              <span className="count">Max {opts?.max_secret_bytes ?? 96} bytes</span>
            </label>
            <input
              value={secret}
              onChange={(e) => setSecret(e.target.value)}
              maxLength={opts?.max_secret_bytes ?? 96}
              placeholder="Enter distributed secret..."
            />
          </div>

          <div className="form-group">
            <label className="form-label">Passphrase</label>
            <input
              value={pass}
              onChange={(e) => setPass(e.target.value)}
              placeholder="Secret encryption key..."
            />
          </div>

          {planInfo && (
            <div className="form-group" style={{ justifyContent: "center" }}>
              <div
                style={{
                  padding: "10px 14px",
                  borderRadius: "var(--radius-md)",
                  background: "var(--bg-input)",
                  border: "1px solid var(--border-subtle)",
                  fontSize: 12.5,
                  color: "var(--text-secondary)",
                }}
              >
                💡 <b>Plan Summary:</b> {planInfo.description}
              </div>
            </div>
          )}
        </div>

        <div className="action-row">
          <button className="btn-primary" onClick={hide} disabled={!!busy || !clip || !opts}>
            {busy ? (
              <>
                <span className="api-pulse" />
                <span>{busy}</span>
              </>
            ) : (
              <span>🛡️ Encode & Disperse across 3 Carriers</span>
            )}
          </button>
          <span className="action-hint">
            Generates synchronous stego audio, stego video, and still image carriers.
          </span>
        </div>

        {err && <div className="alert-box error">⚠️ {err}</div>}
      </section>

      {job && (
        <section className="panel-card">
          <div className="panel-header">
            <div>
              <div className="panel-title">
                <span className="step-tag">Dispersal Status</span>
                <span>Active Plan: {job.k}-of-{job.n} ("{job.plan}")</span>
              </div>
              <p className="panel-desc">
                Any <b>{job.k}</b> of the <b>{job.n}</b> pieces can rebuild the secret. Largest carrier piece: <b>{job.bits_per_carrier} bits</b> (total <b>{job.bits_total} bits</b>).
              </p>
            </div>
          </div>

          {/* Allocation split bar (same look as Audio + Video tab) */}
          {(() => {
            const bb = job.bits_by_carrier || {};
            const keys = job.carriers.filter((k) => bb[k] != null);
            const total = keys.reduce((a, k) => a + bb[k], 0) || 1;
            const colors = { audio: "var(--carrier-audio)", video: "var(--carrier-video)", image: "var(--carrier-image)" };
            const names = { audio: "Audio Stream", video: "Video Stream", image: "Image Still" };
            return (
              <div className="carrier-split-container">
                <div className="carrier-bar">
                  {keys.map((k) => (
                    <div key={k} style={{ width: `${(bb[k] / total) * 100}%`, background: colors[k], transition: "width 0.4s ease" }}
                         title={`${names[k]}: ${bb[k]} bits (${((bb[k] / total) * 100).toFixed(1)}%)`} />
                  ))}
                </div>
                <div className="carrier-legend">
                  {keys.map((k) => (
                    <span key={k} className="legend-item">
                      <span className="legend-dot" style={{ background: colors[k] }} />
                      <span>{names[k]}: <b>{bb[k]} bits</b> ({((bb[k] / total) * 100).toFixed(1)}%)</span>
                    </span>
                  ))}
                </div>
              </div>
            );
          })()}

          {job.risk_driven && (
            <div className="carrier-split-container">
              <div className="metric-title" style={{ marginBottom: 8 }}>
                Risk measured on this clip (detector score with 128 bits hidden alone; above 0.5 means flagged)
              </div>
              <table className="data-table">
                <thead>
                  <tr><th>Carrier</th><th>Risk @128 bits</th><th>Safe capacity</th><th>Piece size</th></tr>
                </thead>
                <tbody>
                  {job.carriers.map((k) => (
                    <tr key={k}>
                      <td><b>{k.toUpperCase()}</b></td>
                      <td>{job.risk_at_128_bits?.[k] == null ? "-" : fmt(job.risk_at_128_bits[k], 3)}</td>
                      <td>{job.safe_capacity_bits?.[k] ?? "-"} bits</td>
                      <td>{job.bytes_by_carrier?.[k] ?? "-"} bytes ({fmt((job.shares?.[k] ?? 0) * 100)}%)</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="panel-desc" style={{ marginTop: 8 }}>
                The more a carrier is flagged, the smaller its piece. All three pieces are needed to rebuild the secret.
              </p>
              {job.risk_note && <div className="alert-box error">⚠️ {job.risk_note}</div>}
            </div>
          )}

          <div className="metrics-grid">
            {job.quality.snr_db != null && (
              <div className="metric-card">
                <div className="metric-val">{fmt(job.quality.snr_db)} <span style={{ fontSize: 13, color: "var(--text-tertiary)" }}>dB</span></div>
                <div className="metric-title">Audio SNR</div>
                <div className="metric-desc">Acoustic fidelity</div>
              </div>
            )}
            {job.quality.video_psnr_db != null && (
              <div className="metric-card">
                <div className="metric-val">{fmt(job.quality.video_psnr_db)} <span style={{ fontSize: 13, color: "var(--text-tertiary)" }}>dB</span></div>
                <div className="metric-title">Video PSNR</div>
                <div className="metric-desc">SSIM: {fmt(job.quality.video_ssim, 3)}</div>
              </div>
            )}
            {job.quality.image_psnr_db != null && (
              <div className="metric-card">
                <div className="metric-val">{fmt(job.quality.image_psnr_db)} <span style={{ fontSize: 13, color: "var(--text-tertiary)" }}>dB</span></div>
                <div className="metric-title">Image PSNR</div>
                <div className="metric-desc">Still frame transparency</div>
              </div>
            )}
          </div>

          <div className="download-links">
            {job.carriers.map((k) => (
              <a key={k} className="download-btn" href={`${API}/download3/${job.job_id}/${k}`} download>
                ⬇️ Download {k.toUpperCase()} Carrier
              </a>
            ))}
          </div>

          {/* Attack Simulator */}
          <h3 style={{ fontSize: 15, marginTop: 24, marginBottom: 8, color: "var(--text-primary)" }}>
            Simulate Loss of Entire Modality
          </h3>
          <p className="panel-desc" style={{ marginBottom: 14 }}>
            Test channel survivability under intentional carrier degradation ("kill_video", "kill_audio", etc.).
          </p>

          <div className="form-grid">
            <div className="form-group">
              <label className="form-label">Attack Scenario</label>
              <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
                {opts.scenarios.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="action-row">
            <button className="btn-secondary" onClick={attack} disabled={!!busy}>
              {busy ? "Testing Attack..." : "⚡ Execute Attack & Extract"}
            </button>
            <button className="btn-ghost" onClick={sweep} disabled={!!busy}>
              🧪 Sweep All Attack Scenarios
            </button>
          </div>

          {res && (
            <div className="extract-result-box" style={{ marginTop: 18 }}>
              <div className="extract-header">
                <div>
                  <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-secondary)" }}>
                    Scenario Result: <b>{res.scenario}</b> &bull;{" "}
                  </span>
                  <span className={`status-chip ${res.recovered ? "clean" : "flagged"}`}>
                    {res.recovered ? "✓ Secret Recovered" : "✗ Reconstruction Failed"}
                  </span>
                </div>
              </div>

              <div style={{ margin: "10px 0" }}>
                {Object.entries(res.carrier_status).map(([k, v]) => (
                  <StatusChip key={k} name={k.toUpperCase()} state={v} />
                ))}
              </div>

              <div className="secret-display">
                {res.recovered
                  ? res.secret
                  : `Failed to recover. Required ${res.needed} readable carriers out of ${res.of}.`}
              </div>
            </div>
          )}

          {table && (
            <div style={{ marginTop: 24 }}>
              <h4 style={{ fontSize: 14, color: "var(--text-primary)", marginBottom: 8 }}>
                Scenario Resilience Matrix
              </h4>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Scenario</th>
                    <th>Audio Status</th>
                    <th>Video Status</th>
                    <th>Image Status</th>
                    <th>Extraction Outcome</th>
                  </tr>
                </thead>
                <tbody>
                  {table.map((r) => (
                    <tr key={r.scenario}>
                      <td><code>{r.scenario}</code></td>
                      {["audio", "video", "image"].map((k) => {
                        const isOk = r.carrier_status[k] === "ok";
                        return (
                          <td key={k}>
                            <span className={`status-chip ${isOk ? "clean" : "flagged"}`}>
                              {isOk ? "Surviving" : "Destroyed"}
                            </span>
                          </td>
                        );
                      })}
                      <td>
                        <span className={`status-chip ${r.recovered ? "clean" : "flagged"}`}>
                          {r.recovered ? "✓ Recovered" : "✗ Lost"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
