"""做法A 真·泛化压力测试：训Core.W + 大数据 + D64/D256不同测试集

关键改动：
1. Core.W 参与训练（跟 emb+ro 一起被 Adam 优化）
2. D=64 和 D=256 用不同的测试数据量——D=256 的测试集更大
3. 逐步加数据直到 D=64 过拟合/饱和，看 D=256 能否继续涨
4. 控制训练样本数迫使泛化，不靠背诵
"""
import torch, torch.nn as nn, torch.nn.functional as F
import sys, os, json, numpy as np, random, time, argparse
from collections import Counter

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'runs', 'transplant_stress')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Data: 5x more than before ───
DATA_PATH = os.path.join(SCRIPT_DIR, 'sample_data.txt')
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw = f.read(300_000)
cnt = Counter(raw)
all_chars = sorted([c for c, n in cnt.items() if n >= 5 and '一' <= c <= '鿿'])
mid = len(all_chars) // 2
TRAIN_CHARS = all_chars[:mid]
TEST_CHARS  = all_chars[mid:]

def make_taskAB(vocab_size, char_set, n_ids=80000):
    """Build data with n_ids samples from specified char_set."""
    chars = ['<PAD>', '<UNK>'] + char_set[:vocab_size]
    c2i = {c: i for i, c in enumerate(chars)}
    ids = [c2i.get(c, 1) for c in raw if '一' <= c <= '鿿'][:n_ids]
    Xs, Ya, Yb = [], [], []
    for i in range(len(ids) - 6):
        Xs.append(ids[i:i+4]); Ya.append(ids[i+4]); Yb.append(ids[i+5])
    return (torch.tensor(Xs), torch.tensor(Ya), torch.tensor(Yb), len(chars))

# ─── Model (same as before) ───
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

# ─── A* ───
def fast_A_det(core, h, n=40):
    for _ in range(10): h = core.step_det(h)
    tr = []
    for _ in range(n):
        h = core.step_det(h); tr.append(h/(h.norm(dim=-1,keepdim=True)+1e-8))
    ds = [0.5*(tr[i]-tr[i+1]).norm(dim=-1).mean().item() for i in range(len(tr)-1)]
    S = 1-min(np.mean(ds),1.); return 4*(1-S)*S/0.3
def _core_step_det(self, h, x=None):
    hh = h.detach()
    ext = self.inp(x) if x is not None else torch.zeros_like(hh)
    return 0.9 * torch.tanh(hh @ self.W.T + self.b + ext) + 0.1 * self.g * hh
Core.step_det = _core_step_det

# ─── Train WITH Core.W ───
def train_eval_trainableW(core, vocab, n_train=200, n_train_steps=400, B=32, char_set=TRAIN_CHARS):
    """Train emb+ro+Core.W on char_set, test on TEST_CHARS (generalization)."""
    d = core.W.shape[0]
    X_tr, YA_tr, YB_tr, V = make_taskAB(vocab, char_set)
    X_te, YA_te, YB_te, _ = make_taskAB(vocab, TEST_CHARS)

    emb = nn.Embedding(V, d//4).to(DEV); ro = nn.Linear(d, V).to(DEV)
    # TRAIN Core.W too!
    opt = torch.optim.Adam(list(emb.parameters())+list(ro.parameters())+list(core.parameters()), lr=0.005)

    loss_curve = []
    # Train on domain A
    for step in range(n_train_steps):
        # Random batch from train domain
        idx = torch.randint(0, len(X_tr), (B,))
        bx, by = X_tr[idx].to(DEV), YA_tr[idx].to(DEV)
        e = emb(bx).flatten(1); hh = torch.zeros(B, d, device=DEV)
        for _ in range(3): hh = core.step(hh, e)
        loss = F.cross_entropy(ro(hh), by)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_curve.append(float(loss.item()))

    # Evaluate on both train domain and test domain
    @torch.no_grad()
    def acc(X, Y):
        c, t = 0, 0
        for i in range(0, min(len(X), 500), B):
            bx, by = X[i:i+B].to(DEV), Y[i:i+B].to(DEV)
            e = emb(bx).flatten(1); hh = torch.zeros(len(bx), d, device=DEV)
            for _ in range(3): hh = core.step(hh, e)
            c += (ro(hh).argmax(-1)==by).sum().item(); t += len(by)
        return c/t

    train_acc = acc(X_tr, YA_tr)
    test_acc = acc(X_te, YA_te)  # generalization to unseen chars!

    # W spectrum after training
    W_np = core.W.detach().cpu().numpy()
    eigs = np.linalg.eigvals(W_np); U,S,Vh = np.linalg.svd(W_np, full_matrices=False)
    abs_eigs = np.abs(eigs); s_norm = S/(S.sum()+1e-8)
    participation = 1.0/((s_norm**2).sum()+1e-8)

    return {"train_acc": train_acc, "test_acc": test_acc,
            "final_loss": loss_curve[-1] if loss_curve else None,
            "loss_curve": loss_curve,
            "sr_post": float(np.max(abs_eigs)),
            "eff_rank_post": int(np.sum(S>0.01*S[0])),
            "participation_post": float(participation),
            "n_train_steps": n_train_steps,
            "vocab": vocab}

# ─── GA: find best D=64 genome (with trainable W in eval!) ───
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
        scores,As,test_accs,hs=[],[],[],[]
        for ind in pop_list:
            core=ind.build()
            h=torch.randn(4,64,device=DEV)*0.5; A_star=fast_A_det(core,h,40)
            if A_star<1.0: scores.append(-1); As.append(A_star); test_accs.append(0); hs.append(999); continue
            res=train_eval_trainableW(core,vocab,n_train_steps=200)
            test_acc=res["test_acc"]
            hc=ind.sr*64*(1-ind.sp)+ind.gain*10
            lp=0.003*max(hc-30,0)+0.01*max(8-hc,0)
            scores.append(test_acc-lp); As.append(A_star); test_accs.append(test_acc); hs.append(hc)
        alive=sum(1 for a in As if a>=1.0)/pop*100; med_test=np.median(test_accs)
        med_h=np.median([h for h in hs if h<900]); bi=np.argmax(scores)
        if scores[bi]>best_fit: best_fit=scores[bi]; best_ind=pop_list[bi]
        fit_cutoff=np.percentile([s for s in scores if s>=0],30)
        lean_hi,lean_lo=np.percentile(hs,70),np.percentile(hs,10)
        keep=[i for i,s in enumerate(scores) if As[i]>=1.0 and s>=fit_cutoff and lean_lo<=hs[i]<=lean_hi]
        if len(keep)<pop*0.2: keep=list(range(pop))
        if alive>30 and med_test>0.5 and 10<med_h<40: vocab=min(30,vocab+3)
        elif alive<10 or med_test<0.2: vocab=max(10,vocab-3)
        survivors=[pop_list[i] for i in keep]
        new_pop=[pop_list[i] for i in keep[:max(1,len(keep)//4)]]
        while len(new_pop)<pop:
            if len(survivors)<2: break
            p1,p2=random.sample(survivors,2); c=Genome.crossover(p1,p2); c.mutate(0.4); new_pop.append(c)
        while len(new_pop)<pop: new_pop.append(Genome())
        pop_list=new_pop
    return best_ind,best_fit

# ══════════════════════════════════════════════
# STRESS TEST
# ══════════════════════════════════════════════
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--target_d",type=int,default=256)
    p.add_argument("--vocabs",type=str,default="30,100,200,300,500")
    p.add_argument("--train_steps",type=str,default="100,200,400,800")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--skip_ga",action="store_true")
    p.add_argument("--n_copies",type=int,default=3)
    args=p.parse_args()

    vocabs=[int(v) for v in args.vocabs.split(",")]
    train_step_list=[int(s) for s in args.train_steps.split(",")]
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    # Get best D=64 genome
    genome_path=os.path.join(OUT_DIR,'best_genome_D64.json')
    if args.skip_ga and os.path.exists(genome_path):
        with open(genome_path) as f: gd=json.load(f)
        best=Genome(); best.sr=gd['sr']; best.gain=gd['gain']; best.sp=gd['sp']
        print(f"Loaded: sr={best.sr:.4f} gain={best.gain:.4f} sp={best.sp:.4f}")
    else:
        print("GA search (trainable W, Mountain 1 generalization)..."); best,_=evolve_best(args.seed)
        with open(genome_path,'w') as f: json.dump(best.to_dict(),f,indent=2)
        print(f"Best: sr={best.sr:.4f} gain={best.gain:.4f} sp={best.sp:.4f}")

    all_rows=[]
    print(f"\n{'v':>4s} {'steps':>5s} {'D64_train':>9s} {'D64_test':>9s} {'D256_train':>9s} {'D256_test':>9s} {'ratio':>7s} {'D64_sr':>7s} {'D256_sr':>7s} {'verdict':>10s}")
    print("-"*100)

    for vocab in vocabs:
        for n_steps in train_step_list:
            # D=64 native
            core64=best.build(ws=args.seed)
            h=torch.randn(4,64,device=DEV)*0.5; fast_A_det(core64,h,40)
            res64=train_eval_trainableW(core64,vocab,n_train_steps=n_steps)

            # D=256 transplant (avg over copies)
            d256_train_accs=[]; d256_test_accs=[]; d256_srs=[]
            for ci in range(args.n_copies):
                tcore=Core(args.target_d,best.sr,best.gain,best.sp,w_seed=args.seed+ci*100).to(DEV)
                res_t=train_eval_trainableW(tcore,vocab,n_train_steps=n_steps)
                d256_train_accs.append(res_t["train_acc"])
                d256_test_accs.append(res_t["test_acc"])
                d256_srs.append(res_t["sr_post"])

            d256_train=np.mean(d256_train_accs); d256_test=np.mean(d256_test_accs)
            d256_sr=np.mean(d256_srs)
            ratio=d256_test/(res64["test_acc"]+1e-8)

            # Verdict: D256 must beat D64 significantly, not just tie
            if res64["test_acc"]>0.95 and d256_test>0.95: verdict="CEILING"
            elif ratio>=1.2: verdict="D256_WINS"  # D256 比 D64 好 20%+
            elif ratio>=0.95: verdict="TIED"
            elif ratio>=0.8: verdict="PASS"
            else: verdict="FAIL"

            row={"vocab":vocab,"n_steps":n_steps,
                 "D64_train":res64["train_acc"],"D64_test":res64["test_acc"],
                 "D64_sr_post":res64["sr_post"],"D64_eff_rank":res64["eff_rank_post"],
                 "D256_train":d256_train,"D256_test":d256_test,"D256_sr_post":d256_sr,
                 "ratio":ratio,"verdict":verdict}
            all_rows.append(row)

            print(f"{vocab:>4d} {n_steps:>5d} {res64['train_acc']:>9.4f} {res64['test_acc']:>9.4f} "
                  f"{d256_train:>9.4f} {d256_test:>9.4f} {ratio:>7.3f} "
                  f"{res64['sr_post']:>7.3f} {d256_sr:>7.3f} {verdict:>10s}")

    # Save
    result={"config":{"target_d":args.target_d,"vocabs":vocabs,"train_steps":train_step_list,
                      "seed":args.seed,"genome":best.to_dict(),"note":"Core.W IS trained"},
            "rows":all_rows}
    with open(os.path.join(OUT_DIR,'result.json'),'w') as f: json.dump(result,f,indent=2,default=float)

    # Summary
    lines=[f"=== 做法A 真泛化压力测试 ===",
           f"Genome: sr={best.sr:.4f} gain={best.gain:.4f} sp={best.sp:.4f}",
           f"Core.W trained with emb+ro | Mountain 1 generalization",
           f"Target D: {args.target_d} | D256 wins if test_acc ratio >= 1.2",
           f"","Grid:",""]
    for r in all_rows:
        lines.append(f"  v={r['vocab']:>4d} steps={r['n_steps']:>4d}  "
                     f"D64_test={r['D64_test']:.4f}  D256_test={r['D256_test']:.4f}  "
                     f"ratio={r['ratio']:.3f}  {r['verdict']}")
    with open(os.path.join(OUT_DIR,'summary.txt'),'w',encoding='utf-8') as f: f.write('\n'.join(lines))
    print(f"\nSaved: {OUT_DIR}")

if __name__=='__main__': main()
