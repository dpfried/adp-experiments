import glob, sys, statistics, itertools
from math import comb
sys.path.insert(0,"/tmp")
from load_rollouts import load_cell, read_jsonl
SE="/checkpoint/dpf/swebench-eval"
OFFERED={"TerminalAction","FileEditorAction","ThinkAction","FinishAction","TaskTrackerAction"}

def feats(r):
    h=r.get("history") or []
    ae=[e for e in h if e.get("kind")=="ActionEvent"]
    if not ae: return None
    t=sum(1 for e in ae if (e.get("action") or {}).get("kind")=="ThinkAction")
    b=sum(1 for e in ae if ((e.get("action") or {}).get("kind") or "") not in OFFERED)
    ag=sum(1 for e in h if e.get("kind")=="AgentErrorEvent")
    narr=sum(1 for e in ae if (lambda v: "".join(x.get("text","") for x in v if isinstance(x,dict)) if isinstance(v,list) else (v or ""))(e.get("thought")).strip())
    return dict(n=len(ae), think=100*t/len(ae), bad=100*b/len(ae), agerr=ag,
                narr=100*narr/len(ae))

E={}
for sh in ["00","01"]:
    c=load_cell(f"{SE}/runs/out_par_E_base_stock_evalp__s{sh}",
                select=f"{SE}/select/shard_{sh}of10.txt", attempt=1)
    for r in c.rows:
        f=feats(r)
        if f: E[r["instance_id"]]=f
G={}
for sh in ["00","01"]:
    p=glob.glob(f"{SE}/runs/out_par_G_base_prefill_evalp__s{sh}/*/*/*/output.jsonl")
    if not p: continue
    seen={}
    for r in read_jsonl(p[0])[0]: seen.setdefault(r.get("instance_id"),r)
    for iid,r in seen.items():
        f=feats(r)
        if f: G[iid]=f

both=sorted(set(E)&set(G))
print(f"G transcripts={len(G)}  E transcripts={len(E)}  PAIRED on both={len(both)}")
print(f"  (G-only, i.e. E has no transcript for them: {len(set(G)-set(E))} -- E capped/errored there)")

def sign_test(pairs):
    """two-sided exact sign test on non-tied pairs"""
    pos=sum(1 for a,b in pairs if b>a); neg=sum(1 for a,b in pairs if b<a)
    n=pos+neg
    if n==0: return pos,neg,1.0
    k=min(pos,neg)
    p=min(1.0, 2*sum(comb(n,i) for i in range(k+1))/2**n)
    return pos,neg,p

print(f"\n{'metric':<26} {'E (paired)':>12} {'G (paired)':>12} {'G>E':>5} {'G<E':>5} {'sign p':>8}")
for key,lab in [("narr","narration %"),("think","ThinkAction %"),
                ("bad","malformed tool %"),("agerr","AgentErrorEvents"),("n","action events")]:
    pairs=[(E[i][key],G[i][key]) for i in both]
    me=statistics.median([a for a,_ in pairs]); mg=statistics.median([b for _,b in pairs])
    pos,neg,p=sign_test(pairs)
    print(f"{lab:<26} {me:12.2f} {mg:12.2f} {pos:5} {neg:5} {p:8.4f}")
