EPS_NORMALIZE=1e-12
CLIP_NORM=5.
CLIP_EPS=1e-6

def normalize_capture(hidden):
    
    
    norm=hidden.norm(p=2,dim=1,keepdim=True)
    denominator=norm.clamp_min(EPS_NORMALIZE).expand_as(hidden)
    z=hidden/denominator
    return z,norm.detach()>=EPS_NORMALIZE

def channel_measurement(z,boundary,unclamped,raw_lambda):
    product=boundary.detach().double()*raw_lambda.detach().double()
    row_inner=product.sum(1,keepdim=True)
    return (product-z.detach().double().square()*row_inner*unclamped).sum(0)

def euclidean(n,difference_half):
    import torch
    square=n.square().sum(0);nonzero=square!=0
    factor=torch.where(nonzero,difference_half/torch.where(nonzero,square,torch.ones_like(square)),torch.zeros_like(square))
    return n*factor.unsqueeze(0),(~nonzero).sum().double()

def maximal_tau(full_q_square,q_selected,r,cap=CLIP_NORM):
    import torch
    a=r.square().sum();b=(q_selected*r).sum().abs();raw_slack=cap*cap-full_q_square
    slack=raw_slack.clamp_min(0.)
    denominator=(b.square()+a*slack).sqrt()+b
    safe=torch.where(denominator>0,denominator,torch.ones_like(denominator))
    candidate=(slack/safe).clamp(max=1.)
    
    
    
    tau=torch.where(a==0,torch.ones_like(a),torch.where(raw_slack>0,candidate,torch.zeros_like(a)))
    return tau,(-raw_slack).clamp_min(0.),(raw_slack<0).double()

def specification():
    return dict(S=['conv_layers.2.params.2','conv_layers.2.params.1'],S_objects=['aggregators[2].scale[0]','aggregators[2].offset[0]'],
        measurement='mu_tj=sum_rows(v_j*lambda_tj-I_native_unclamped*z_j^2*sum_k(v_k*lambda_tk))',
        identity='mu_tj=gamma_j*g_gamma_tj+beta_j*g_beta_tj; native weighted BCE-sum reduction is already in raw lambda.',
        normalization='Original Tensor.norm -> clamp_min(1e-12) -> expand_as -> divide. Equality uses unclamped derivative; zero rows use clamped branch.',
        recovery='Raw Euclidean r_j=n_j*(mu1-mu0)_j/(2*n_j^T*n_j); n=(gamma,beta), exact-zero n maps to zero.',
        mean_clip='Original native P/L/C clip_grad_norm_(5), epsilon1e-6 unchanged. r receives the actual same L mean-clip coefficient.',
        tau='Largest tau in [0,1] with ||q_L +/- tau*r||<=5 in real arithmetic, using all original L coordinates including uncorrected W/b.',
        retained_measurement='Final half-difference retains tau*kappa_L*(mu1-mu0)/2; tau<1 does not preserve the full raw measurement. No separate native clip is applied after forming the pair.',
        native_rounding_fallback='For nonzero r and a native postclip q whose FP64 norm exceeds5, tau=0 keeps original q/q and records infeasible native-rounding domain. r=0 uses tau1. No reclip, arbitrary epsilon or mean shift.',
        precision='FP64 measurement/projection/geometry, final selected gradients cast to original dtype. Real-arithmetic mean/cap identities are not bitwise FP32 guarantees; postcast mean error/cap excess recorded.',
        original_three_API='Head1; Head2+Body1; Body2',extra_Body_forwards=0,extra_Body_VJPs=0,extra_SpMM=0,
        cost='Retained z/branch, two O(nodes*512) contractions, Euclidean decode, whole-L norm, tau/gradient install and device statistics are paid; one small epoch-boundary flush is inside train_epoch T1.')
