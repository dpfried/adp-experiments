import sys, glob
sys.path.insert(0, "/checkpoint/dpf/swebench-eval/analysis")
from load_rollouts import load_cell, RolloutIntegrityError

SEL="/checkpoint/dpf/swebench-eval/select/shard_{n}of10.txt"
CELLS={
 "E_base_stock":  "/checkpoint/dpf/swebench-eval/runs/out_par_E_base_evalp__{sh}",
 "G_base_prefill":"/checkpoint/dpf/swebench-eval/runs/out_par_G_base_prefill_evalp__{sh}",
}
for name,tpl in CELLS.items():
    tot={}
    for sh,n in (("s00","00"),("s01","01")):
        d=tpl.format(sh=sh)
        if not glob.glob(d): print(f"{name} {sh}: MISSING"); continue
        try:
            c=load_cell(d, select=SEL.format(n=n), attempt=1, strict=False)
        except Exception as e:
            print(f"{name} {sh}: ERR {type(e).__name__}: {e}"); continue
        t=c.taxonomy
        print(f"{name} {sh}  src={c.source}  n={len(c.ids)}  patch={len(c.gradeable)}  transcript={len(c.transcript_ids)}")
        print(f"    {dict(sorted(t.items()))}")
        for k,v in t.items(): tot[k]=tot.get(k,0)+v
        tot["_n"]=tot.get("_n",0)+len(c.ids); tot["_patch"]=tot.get("_patch",0)+len(c.gradeable)
        tot["_tr"]=tot.get("_tr",0)+len(c.transcript_ids)
    n=tot.pop("_n",0); p=tot.pop("_patch",0); tr=tot.pop("_tr",0)
    if n: print(f"  == {name} POOLED: n={n} patch={p} transcript={tr} | " +
                "  ".join(f"{k}={v} ({100*v/n:.1f}%)" for k,v in sorted(tot.items())))
    print()
