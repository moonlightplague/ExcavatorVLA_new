import json, sys, math

path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]
print(f'total_steps={len(rows)}')
print('step bucket_deg open scooped retained spillage payload | unload_fwd unload_left unload_up')
prev = None
for r in rows:
    m = r['particle_transfer_metrics']
    bp = r['bucket_position']
    bp_deg = bp * 180 / math.pi
    bucket_open = bp >= -0.80
    scooped = m['scooped_particles']
    retained = m['retained_particles']
    spillage = m['spillage_particles']
    payload = m['payload_particles']
    s = r['input_state']
    unload = s[21:24]  # unload_from_load_forward/left/up
    uf, ul, uu = unload
    # 只在关键转折点打印（scooped/payload/bucket_open 变化，或每 20 步）
    key = (scooped, payload, bucket_open)
    if key != prev or r['step'] % 20 == 0:
        print(f"{r['step']:4d} {bp_deg:8.1f} {str(bucket_open):5s} {scooped:6d} {retained:6d} {spillage:6d} {payload:6d} | {uf:7.2f} {ul:7.2f} {uu:7.2f}")
        prev = key
