"""One-time patch: makes api/three.py log to SQLite. Safe to run twice."""
from pathlib import Path
p = Path("api/three.py")
s = p.read_text(encoding="utf-8")
if "from api.db import" in s:
    print("already patched"); raise SystemExit
s = s.replace("from api.main import OUT, ROOT, clean, load_clip\n",
              "from api import db\nfrom api.db import history as _history\nfrom api.main import OUT, ROOT, clean, load_clip\n", 1)
a = '    return {\n        "job_id": job, "plan": req.plan,'
assert a in s, "embed3 return not found"
s = s.replace(a, '    out = {\n        "job_id": job, "plan": req.plan,', 1)
b = '            "image_psnr_db": clean(psnr(still[None], si[None])) if "image" in used else None},\n    }\n'
assert b in s, "embed3 end not found"
s = s.replace(b, b + '    db.log_job(out, req.clip_id)\n    return out\n', 1)
c = '    return {"recovered": secret is not None, "scenario": req.scenario,'
assert c in s, "extract3 return not found"
s = s.replace(c, '    res = {"recovered": secret is not None, "scenario": req.scenario,', 1)
d = '            "secret": secret.decode(errors="replace") if secret is not None else None}\n'
assert d in s, "extract3 end not found"
s = s.replace(d, d + '    db.log_extract(req.job_id, req.scenario, res["recovered"], res["carrier_status"])\n    return res\n\n\n'
              '@router.get("/history3")\ndef history3(limit: int = 50):\n    return _history(limit)\n', 1)
p.write_text(s, encoding="utf-8")
print("patched OK")
