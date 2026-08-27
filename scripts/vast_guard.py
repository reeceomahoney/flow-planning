import sys

from vastai import VastAI

MAX_DPH = 1.0
GPU = "RTX 5090"

rows = VastAI().show_instances()
rows = rows if isinstance(rows, list) else rows.get("instances", [])
bad = [
    r
    for r in rows
    if r.get("num_gpus") != 1
    or GPU not in str(r.get("gpu_name"))
    or float(r.get("dph_total", 9)) > MAX_DPH
]
for r in rows:
    print(
        f"instance {r.get('id')}: {r.get('gpu_name')} x{r.get('num_gpus')} "
        f"${float(r.get('dph_total', 0)):.3f}/h {r.get('geolocation')}"
    )
if bad:
    print(
        f"GUARD: {len(bad)} instance(s) violate 1x {GPU} <= ${MAX_DPH}/h",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"GUARD ok: {len(rows)} instance(s), all 1x {GPU} <= ${MAX_DPH}/h")
