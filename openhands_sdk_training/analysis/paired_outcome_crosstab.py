import sys, glob, collections
sys.path.insert(0, "/checkpoint/dpf/swebench-eval/analysis")
from load_rollouts import load_cell
R="/checkpoint/dpf/swebench-eval/runs/"
SEL="/checkpoint/dpf/swebench-eval/select/shard_{n}of10.txt"
E={}; G={}
for sh,n in (("s00","00"),("s01","01")):
    for d,acc in ((f"{R}out_par_E_base_stock_evalp__{sh}",E),(f"{R}out_par_G_base_prefill_evalp__{sh}",G)):
        c=load_cell(d, select=SEL.format(n=n), attempt=1, strict=False)
        for i in c.ids: acc[i]=c
        acc.setdefault("_cells",[]).append(c)
Ecells=E.pop("_cells"); Gcells=G.pop("_cells")
from load_rollouts import classify
Ecl={i:classify(c.by_id[i]) for c in Ecells for i in c.ids}
Gcl={i:classify(c.by_id[i]) for c in Gcells for i in c.ids}
Epatch={i for c in Ecells for i in c.gradeable}; Gpatch={i for c in Gcells for i in c.gradeable}
common=sorted(set(Ecl)&set(Gcl))
print(f"G has attempted {len(Gcl)}/100 ; paired subset = {len(common)}\n")
print("=== PAIRED outcome-class crosstab (rows E, cols G) on G's completed subset ===")
ks=sorted({Ecl[i] for i in common}|{Gcl[i] for i in common})
ct=collections.Counter((Ecl[i],Gcl[i]) for i in common)
hdr="  E \\ G     "+" ".join(f"{k:>9}" for k in ks)+"   | tot"
print(hdr); print("  "+"-"*(len(hdr)-2))
for a in ks:
    row=[ct[(a,b)] for b in ks]
    print(f"  {a:<10} "+" ".join(f"{v:>9}" for v in row)+f"   | {sum(row)}")
print("  "+"-"*(len(hdr)-2))
print(f"  {'tot':<10} "+" ".join(f"{sum(ct[(a,b)] for a in ks):>9}" for b in ks))
print()
def rate(cl,ids,k): 
    n=sum(1 for i in ids if cl[i]==k); return n,100*n/len(ids)
for k in ("cap500","ok","stuck"):
    en,ep=rate(Ecl,common,k); gn,gp=rate(Gcl,common,k)
    print(f"  {k:<8} E {en:>3}/{len(common)} ({ep:5.1f}%)   G {gn:>3}/{len(common)} ({gp:5.1f}%)")
ep=len(Epatch&set(common)); gp=len(Gpatch&set(common))
print(f"  {'patch':<8} E {ep:>3}/{len(common)} ({100*ep/len(common):5.1f}%)   G {gp:>3}/{len(common)} ({100*gp/len(common):5.1f}%)")
# McNemar on cap500 and on patch-presence
def mcn(a,b,label):
    n01=sum(1 for i in common if a(i) and not b(i)); n10=sum(1 for i in common if b(i) and not a(i))
    from math import comb
    n=n01+n10; k=min(n01,n10)
    p=min(1.0, 2*sum(comb(n,j) for j in range(k+1))/2**n) if n else 1.0
    print(f"  McNemar {label}: E-only={n01} G-only={n10}  exact p={p:.4g}")
mcn(lambda i:Ecl[i]=="cap500", lambda i:Gcl[i]=="cap500", "cap500")
mcn(lambda i:i in Epatch,      lambda i:i in Gpatch,      "has-patch")
