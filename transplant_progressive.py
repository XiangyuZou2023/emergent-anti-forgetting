"""做法A 渐进难度测试：找到 D=64→D=256 移植的失效 vocab

规则：
- 每个 vocab 级别：跑 D=64 native + D=256 transplant（直接移植参数）
- D=256 ≈ D=64（差距<5%）→ 任务太简单，升 vocab
- D=256 < D=64×0.8 → 做法A在此失效，记录断点
- D=64 < 0.3 → 任务对native也太难，停止
- 上限 0.95（天花板效应）

输出：runs/transplant_progressive/result.json + summary.txt
"""
import torch, torch.nn as nn, torch.nn.functional as F
import sys, os, json, numpy as np, random, time, argparse
from collections import Counter

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'runs', 'transplant_progressive')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Data ───
DATA_PATH = os.path.join(SCRIPT_DIR, 'sample_data.txt')
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw = f.read(300_000)
cnt = Counter(raw)
all_chars = sorted([c for c, n in cnt.items() if n >= 5 and '一' <= c <= '鿿'])
mid = len(all_chars) // 2
TRAIN_CHARS = all_chars[:mid]
TEST_CHARS  = all_chars[mid:]

def make_taskAB(vocab_size, char_set):
    chars = ['<PAD>', '<UNK>'] + char_set[:vocab_size]
    c2i = {c: i for i, c in enumerate(chars)}
    ids = [c2i.get(c, 1) for c in raw if '一' <= c <= '鿿'][:20000]
    Xs, Ya, Yb = [], [], []
    for i in range(len(ids) - 6):
        Xs.append(ids[i:i+4]); Ya.append(ids[i+4]); Yb.append(ids[i+5])
    return (torch.tensor(Xs), torch.tensor(Ya), torch.tensor(Yb), len(chars))

# ─── Model ───
class Core(nn.Module):
    def __init__(self, d=64, sr=0.9, gain=0.1, sp=0.5, w_seed=None):
        super().__init__(); self.d = d
        if w_seed is not None: torch.manual_seed(w_seed)
        W = torch.randn(d,d)*0.3; mask = (torch.rand(d,d)>sp).float(); W = W*mask
        s = torch.linalg.norm(W,2)
        self.W = nn.Parameter(W*(sr/(s+1e-8)))
        self.b = nn.Parameter(torch.zeros(d)); self.g = nn.Parameter(torch.tensor(gain))
        self.inp = nn.Linear(d, d, bias=False)

    def step(self, h, x=None):
        h = h.detach(); ext = self.inp(x) if x is not None else torch.zeros_like(h)
        return 0.9*torch.tanh(h@self.W.T+self.b+ext)+0.1*self.g*h+torch.randn_like(h)*0.005

    def step_det(self, h, x=None):
        h = h.detach(); ext = self.inp(x) if x is not None else torch.zeros_like(h)
        return 0.9*torch.tanh(h@self.W.T+self.b+ext)+0.1*self.g*h

# ─── A* gate ───
def fast_A_det(core, h, n=40):
    for _ in range(10): h = core.step_det(h)
    tr = []
    for _ in range(n):
        h = core.step_det(h); tr.append(h/(h.norm(dim=-1,keepdim=True)+1e-8))
    ds = [0.5*(tr[i]-tr[i+1]).norm(dim=-1).mean().item() for i in range(len(tr)-1)]
    S = 1-min(np.mean(ds),1.)
    return 4*(1-S)*S/0.3

# ─── Spectrum ───
def analyze_spectrum(W_np):
    eigs = np.linalg.eigvals(W_np); U,S,Vh = np.linalg.svd(W_np,full_matrices=False)
    abs_eigs = np.abs(eigs)
    s_norm = S/(S.sum()+1e-8); participation = 1.0/((s_norm**2).sum()+1e-8)
    sorted_abs = np.sort(abs_eigs)[::-1]
    gap_01 = sorted_abs[0]-sorted_abs[1] if len(sorted_abs)>1 else 0
    _, V = np.linalg.eig(W_np)
    iprs = [(np.abs(V[:,i])**4).sum()/((np.abs(V[:,i])**2).sum()**2+1e-8) for i in range(V.shape[1])]
    return {"D":W_np.shape[0],"spectral_radius":float(np.max(abs_eigs)),
            "mean_abs_eig":float(np.mean(abs_eigs)),"effective_rank":int(np.sum(S>0.01*S[0])),
            "participation_ratio":float(participation),"spectral_gap_ratio":float(gap_01/(sorted_abs[0]+1e-8)),
            "mean_ipr":float(np.mean(iprs)),"top5_sv":[float(x) for x in S[:5]]}

# ─── Train + Eval (Mountain 1 generalization) ───
def train_eval(core, vocab, n_train=200, B=32):
    d = core.W.shape[0]
    X_tr, YA_tr, YB_tr, V = make_taskAB(vocab, TRAIN_CHARS)
    X_te, YA_te, YB_te, _ = make_taskAB(vocab, TEST_CHARS)
    emb = nn.Embedding(V,d//4).to(DEV); ro = nn.Linear(d,V).to(DEV)
    opt = torch.optim.Adam(list(emb.parameters())+list(ro.parameters()),lr=0.01)

    for X, Y in [(X_tr,YA_tr),(X_tr,YB_tr)]:
        perm = torch.randperm(len(X))[:n_train]
        for i in range(0,len(perm),B):
            idx=perm[i:i+B]; bx,by=X[idx].to(DEV),Y[idx].to(DEV)
            e=emb(bx).flatten(1); hh=torch.zeros(len(bx),d,device=DEV)
            for _ in range(3): hh=core.step(hh,e)
            loss=F.cross_entropy(ro(hh),by); opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def acc(X,Y):
        c,t=0,0
        for i in range(0,len(X),B):
            bx,by=X[i:i+B].to(DEV),Y[i:i+B].to(DEV)
            e=emb(bx).flatten(1); hh=torch.zeros(len(bx),d,device=DEV)
            for _ in range(3): hh=core.step(hh,e)
            c+=(ro(hh).argmax(-1)==by).sum().item(); t+=len(by)
        return c/t

    return {"train_acc":acc(X_tr,YA_tr),"test_acc":acc(X_te,YA_te),"vocab":vocab}

# ─── GA: find best D=64 genome ───
class Genome:
    d=64
    def __init__(self,sr=0.9,gain=0.1,sp=0.5):
        self.sr=np.clip(sr+random.uniform(-0.05,0.05),0.5,2.0)
        self.gain=np.clip(gain+random.uniform(-0.05,0.05),0.0,1.0)
        self.sp=np.clip(sp+random.uniform(-0.1,0.1),0.0,0.9)
    def mutate(self,r=0.3):
        if random.random()<r: self.sr=np.clip(self.sr+random.uniform(-0.2,0.2),0.5,2.0)
        if random.random()<r: self.gain=np.clip(self.gain+random.uniform(-0.2,0.2),0.0,1.0)
        if random.random()<r: self.sp=np.clip(self.sp+random.uniform(-0.2,0.2),0.0,0.9)
    def build(self,ws=None): return Core(self.d,self.sr,self.gain,self.sp,w_seed=ws).to(DEV)
    @staticmethod
    def crossover(a,b):
        c=Genome(); c.sr=a.sr if random.random()<0.5 else b.sr
        c.gain=a.gain if random.random()<0.5 else b.gain
        c.sp=a.sp if random.random()<0.5 else b.sp; return c
    def to_dict(self): return {"sr":self.sr,"gain":self.gain,"sp":self.sp,"d":self.d}

def evolve_best(seed=42,pop=20,gens=30):
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    pop_list=[Genome() for _ in range(pop)]; vocab=10; best_ind,best_fit=None,-99
    for gen in range(gens):
        X_tr,YA_tr,YB_tr,V=make_taskAB(vocab,TRAIN_CHARS)
        X_te,YA_te,YB_te,_=make_taskAB(vocab,TEST_CHARS)
        scores,As,accAs,retAs,hs=[],[],[],[],[]
        for ind in pop_list:
            core=ind.build()
            h=torch.randn(4,64,device=DEV)*0.5; A_star=fast_A_det(core,h,40)
            if A_star<1.0: scores.append(-1); As.append(A_star); accAs.append(0); retAs.append(0); hs.append(999); continue
            emb=nn.Embedding(V,16).to(DEV); ro=nn.Linear(64,V).to(DEV)
            opt=torch.optim.Adam(list(emb.parameters())+list(ro.parameters()),lr=0.01)
            def te(tX,tY,n=200):
                perm=torch.randperm(len(tX))[:n]
                for i in range(0,len(perm),32):
                    idx=perm[i:i+32]; bx,by=tX[idx].to(DEV),tY[idx].to(DEV)
                    e=emb(bx).flatten(1); hh=torch.zeros(len(bx),64,device=DEV)
                    for _ in range(3): hh=core.step(hh,e)
                    loss=F.cross_entropy(ro(hh),by); opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    tidx=torch.randint(0,len(tX),(200,)); bx,by=tX[tidx].to(DEV),tY[tidx].to(DEV)
                    e=emb(bx).flatten(1); hh=torch.zeros(200,64,device=DEV)
                    for _ in range(3): hh=core.step(hh,e)
                    return (ro(hh).argmax(-1)==by).float().mean().item()
            aA=te(X_te,YA_te,200); aB=te(X_te,YB_te,200); aR=te(X_te,YA_te,50)
            w=aA*0.2+aB*0.2+aR*0.6
            hc=ind.sr*64*(1-ind.sp)+ind.gain*10
            lp=0.003*max(hc-30,0)+0.01*max(8-hc,0)
            scores.append(w-lp); As.append(A_star); accAs.append(aA); retAs.append(aR); hs.append(hc)
        alive=sum(1 for a in As if a>=1.0)/pop*100; med_ret=np.median(retAs)
        med_h=np.median([h for h in hs if h<900]); bi=np.argmax(scores)
        if scores[bi]>best_fit: best_fit=scores[bi]; best_ind=pop_list[bi]
        fit_cutoff=np.percentile([s for s in scores if s>=0],30)
        lean_hi,lean_lo=np.percentile(hs,70),np.percentile(hs,10)
        keep=[i for i,s in enumerate(scores) if As[i]>=1.0 and s>=fit_cutoff and lean_lo<=hs[i]<=lean_hi]
        if len(keep)<pop*0.2: keep=list(range(pop))
        if alive>30 and med_ret>0.5 and 10<med_h<40: vocab=min(30,vocab+3)
        elif alive<10 or med_ret<0.2: vocab=max(10,vocab-3)
        survivors=[pop_list[i] for i in keep]
        new_pop=[pop_list[i] for i in keep[:max(1,len(keep)//4)]]
        while len(new_pop)<pop:
            if len(survivors)<2: break
            p1,p2=random.sample(survivors,2); c=Genome.crossover(p1,p2); c.mutate(0.4); new_pop.append(c)
        while len(new_pop)<pop: new_pop.append(Genome())
        pop_list=new_pop
    return best_ind,best_fit

# ══════════════════════════════════════════════
# PROGRESSIVE VOCAB SWEEP
# ══════════════════════════════════════════════
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--target_d",type=int,default=256)
    p.add_argument("--vocabs",type=str,default="30,50,75,100,150,200,300,500")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--skip_ga",action="store_true")
    p.add_argument("--n_copies",type=int,default=3)
    p.add_argument("--n_train",type=int,default=200)
    p.add_argument("--ceiling",type=float,default=0.95)
    p.add_argument("--fail_ratio",type=float,default=0.8,
                   help="D256 < D64*fail_ratio => Method A failed at this vocab")
    p.add_argument("--tie_ratio",type=float,default=0.95,
                   help="D256 > D64*tie_ratio => considered tied, task too easy")
    args=p.parse_args()

    vocabs=[int(v) for v in args.vocabs.split(",")]
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ─── Get best D=64 genome ───
    genome_path=os.path.join(OUT_DIR,'best_genome_D64.json')
    if args.skip_ga and os.path.exists(genome_path):
        with open(genome_path) as f: gd=json.load(f)
        best=Genome(); best.sr=gd['sr']; best.gain=gd['gain']; best.sp=gd['sp']
        print(f"Loaded best genome: sr={best.sr:.4f} gain={best.gain:.4f} sp={best.sp:.4f}")
    else:
        print("Evolving best D=64 genome..."); best,_=evolve_best(args.seed)
        with open(genome_path,'w') as f: json.dump(best.to_dict(),f,indent=2)
        print(f"Best: sr={best.sr:.4f} gain={best.gain:.4f} sp={best.sp:.4f}")

    print(f"\nProgressive vocab sweep: {vocabs}")
    print(f"Target D={args.target_d} | Fail if D256 < {args.fail_ratio}*D64 | Tie if D256 > {args.tie_ratio}*D64 | Ceiling={args.ceiling}")
    print(f"{'vocab':>6s} {'D64_acc':>8s} {'D256_acc':>8s} {'ratio':>7s} {'verdict':>12s} {'A*_D64':>7s} {'A*_D256':>7s} {'D64_sr':>7s} {'D256_sr':>7s}")
    print("-"*90)

    all_rows=[]; stopped=None

    for vi, vocab in enumerate(vocabs):
        row={"vocab":vocab}

        # D=64 native
        core64=best.build(ws=args.seed)
        h=torch.randn(4,64,device=DEV)*0.5; A64=fast_A_det(core64,h,40)
        spec64=analyze_spectrum(core64.W.detach().cpu().numpy())
        res64=train_eval(core64,vocab,n_train=args.n_train)
        d64_acc=res64["test_acc"]
        row["D64"]={"acc":d64_acc,"A_star":float(A64),"sr":spec64["spectral_radius"],
                     "eff_rank":spec64["effective_rank"],"participation":spec64["participation_ratio"]}

        # D=256 transplant (avg over copies)
        d256_accs=[]; d256_As=[]; d256_srs=[]
        for ci in range(args.n_copies):
            tcore=Core(args.target_d,best.sr,best.gain,best.sp,w_seed=args.seed+ci*100).to(DEV)
            h=torch.randn(4,args.target_d,device=DEV)*0.5; At=fast_A_det(tcore,h,40)
            d256_As.append(float(At))
            spec_t=analyze_spectrum(tcore.W.detach().cpu().numpy())
            d256_srs.append(spec_t["spectral_radius"])
            res_t=train_eval(tcore,vocab,n_train=args.n_train)
            d256_accs.append(res_t["test_acc"])

        d256_acc=np.mean(d256_accs); d256_std=np.std(d256_accs)
        d256_A=np.mean(d256_As); d256_sr=np.mean(d256_srs)
        ratio=d256_acc/(d64_acc+1e-8)
        row["D256"]={"acc":d256_acc,"acc_std":d256_std,"A_star":d256_A,"sr":d256_sr}
        row["ratio"]=ratio

        # Verdict
        if d64_acc>=args.ceiling and d256_acc>=args.ceiling:
            verdict="AT_CEILING"
        elif ratio>=args.tie_ratio:
            verdict="TIED"
        elif ratio>=args.fail_ratio:
            verdict="PASS"
        else:
            verdict="FAIL"
        row["verdict"]=verdict

        all_rows.append(row)

        print(f"{vocab:>6d} {d64_acc:>8.4f} {d256_acc:>8.4f} ({d256_std:.4f}) {ratio:>7.3f} {verdict:>12s} "
              f"{A64:>7.3f} {d256_A:>7.3f} {spec64['spectral_radius']:>7.3f} {d256_sr:>7.3f}")

        # Stop conditions
        if verdict=="FAIL":
            stopped=f"Method A failed at vocab={vocab}: D256={d256_acc:.4f} < {args.fail_ratio}*D64={d64_acc:.4f}"
        if d64_acc<0.3:
            if not stopped: stopped=f"Native D=64 too weak at vocab={vocab} (acc={d64_acc:.4f})"
        if d64_acc>=args.ceiling and ratio>=args.tie_ratio:
            pass  # keep going, task too easy for both

    # ─── Save ───
    result={"config":{"target_d":args.target_d,"vocabs":vocabs,"seed":args.seed,
                      "fail_ratio":args.fail_ratio,"tie_ratio":args.tie_ratio,"ceiling":args.ceiling,
                      "genome":best.to_dict()},
            "rows":all_rows,"stopped_reason":stopped}
    with open(os.path.join(OUT_DIR,'result.json'),'w') as f: json.dump(result,f,indent=2,default=float)

    lines=[f"=== 做法A 渐进难度测试 ===",
           f"Genome: sr={best.sr:.4f} gain={best.gain:.4f} sp={best.sp:.4f}",
           f"Target D: {args.target_d} | Fail ratio: {args.fail_ratio} | Tie ratio: {args.tie_ratio}",
           f"","Results:",""]
    for r in all_rows:
        lines.append(f"  vocab={r['vocab']:>4d}  D64={r['D64']['acc']:.4f}  D256={r['D256']['acc']:.4f}  "
                     f"ratio={r['ratio']:.3f}  {r['verdict']}")
    if stopped: lines.append(f"\nSTOP: {stopped}")
    else: lines.append("\nSweep complete, no hard failure found.")
    with open(os.path.join(OUT_DIR,'summary.txt'),'w',encoding='utf-8') as f: f.write('\n'.join(lines))

    print(f"\nSaved: {OUT_DIR}")
    if stopped: print(f"STOP: {stopped}")
    else: print("Sweep complete.")

if __name__=='__main__': main()
