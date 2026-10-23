"""Two-kernel state segment on captured Qwen inputs; no model patch."""
import argparse,json,time,statistics
from pathlib import Path
import torch,torch_npu,triton
import triton.language as tl

parser = argparse.ArgumentParser()
parser.add_argument('--tile', type=int, choices=(8,32), default=32)
parser.add_argument('--output', default='results/recurrent_fused_state_probe.json')
args = parser.parse_args()

@triton.jit
def memory_kernel(S,K,G,M,D:tl.constexpr,V:tl.constexpr,C:tl.constexpr):
 h=tl.program_id(0); tile=tl.program_id(1)
 i=tl.arange(0,D); j=tile*C+tl.arange(0,C)
 s=tl.load(S+h*D*V+i[:,None]*V+j[None,:])
 k=tl.load(K+h*D+i)
 decay=tl.load(G+h)
 d=s*decay
 m=tl.sum(d*k[:,None],axis=0)
 tl.store(M+h*V+j,m)

@triton.jit
def update_kernel(S,K,Q,G,VV,B,M,OUT,NEW,D:tl.constexpr,V:tl.constexpr,C:tl.constexpr):
 h=tl.program_id(0);tile=tl.program_id(1)
 i=tl.arange(0,D);j=tile*C+tl.arange(0,C)
 s=tl.load(S+h*D*V+i[:,None]*V+j[None,:]);k=tl.load(K+h*D+i);q=tl.load(Q+h*D+i)
 decay=tl.load(G+h);beta=tl.load(B+h)
 delta=(tl.load(VV+h*V+j)-tl.load(M+h*V+j))*beta
 n=s*decay+k[:,None]*delta[None,:]
 tl.store(NEW+h*D*V+i[:,None]*V+j[None,:],n)
 o=tl.sum(n*q[:,None],axis=0)
 tl.store(OUT+h*V+j,o)

def prepare(r):
 q,k=r['q'],r['k']
 q=q*torch.rsqrt((q*q).sum(-1,keepdim=True)+1e-6)
 k=k*torch.rsqrt((k*k).sum(-1,keepdim=True)+1e-6)
 q,k,v,b,g=[x[:,0].contiguous().float() for x in (q,k,r['v'],r['beta'],r['g'])]
 return q*(128**-.5),k,v,b,g.exp(),r['state'].float()
def baseline(p):
 q,k,v,b,g,s=p
 d=s*g[:,:,None,None];m=(d*k[:,:,:,None]).sum(-2)
 delta=(v-m)*b[:,:,None];n=d+k[:,:,:,None]*delta[:,:,None,:]
 o=(n*q[:,:,:,None]).sum(-2)
 return o[:,None].to(torch.bfloat16),n
def fused(p):
 q,k,v,b,g,s=p;batch,heads,D,V=s.shape
 m=torch.empty_like(v);o=torch.empty_like(v);n=torch.empty_like(s)
 grid=(batch*heads,V//args.tile)
 memory_kernel[grid](s,k,g,m,D,V,args.tile,enable_fp_fusion=False)
 update_kernel[grid](s,k,q,g,v,b,m,o,n,D,V,args.tile,enable_fp_fusion=False)
 return o[:,None].to(torch.bfloat16),n
def compare(a,b):
 return {'output_max_error':(a[0].float()-b[0].float()).abs().max().item(),'state_max_error':(a[1]-b[1]).abs().max().item(),'output_exact':torch.equal(a[0],b[0]),'state_exact':torch.equal(a[1],b[1])}
def timing(fn):
 for _ in range(3):fn()
 torch.npu.synchronize();s=[]
 for _ in range(10):
  t=time.perf_counter();fn();torch.npu.synchronize();s.append((time.perf_counter()-t)*1000)
 return statistics.median(s)
r=torch.load('results/gdn_real_batch_artifact.pt',map_location='cpu',weights_only=False)
r={k:v.to('npu') for k,v in r.items() if torch.is_tensor(v)}
torch.set_num_threads(8)
p=prepare(r);a=baseline(p);b=fused(p);torch.npu.synchronize()
res={'tile':args.tile,'artifact':'results/gdn_real_batch_artifact.pt','scope':'captured repeated-prompt HF decode; not E100; FP32 working state','correctness':compare(a,b),'input_state_unchanged':torch.equal(p[-1],r['state'].float())}
res['segment_ms']={name:timing(lambda:fn(p)) for name,fn in [('baseline',baseline),('fused',fused)]}
res['wrapper_ms']={name:timing(lambda:fn(prepare(r))) for name,fn in [('baseline',baseline),('fused',fused)]}
print(json.dumps(res,indent=2),flush=True)
Path(args.output).write_text(json.dumps(res,indent=2)+'\n')
