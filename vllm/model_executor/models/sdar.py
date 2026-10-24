# SPDX-License-Identifier: Apache-2.0
"""SDAR block-diffusion LM (JetLM/SDAR-*-Chat) on Model Runner V2.

SDAR is a Qwen3 backbone trained with block-causal ("staircase") attention:
a token sees every earlier block plus its whole own block of B tokens.
Generation fills one B-token canvas of mask tokens at a time:

  denoise step: forward the canvas against the cached prefix, sample every
      position, unmask the masked ones whose sampled-token probability
      exceeds a threshold (at least ``B // steps`` per step, by confidence);
      the canvas KV is thrown away (all canvas tokens "rejected").
  commit step: once no mask is left (or ``steps`` steps ran), forward the
      final canvas once more so its KV is kept, and emit its B tokens.

This reuses the spec-decode data path like DiffusionGemma: the canvas rides
in ``draft_tokens`` and the attention is the Triton backend with the
staircase mask switched on by ``VLLM_BLOCK_CAUSAL=B`` (every canvas starts
on a B-aligned position, so one mask serves prefill, denoise and commit).

Per-request ``SamplingParams.extra_args``:
  sdar_temperature (float, default 1.0), sdar_threshold (default 0.9),
  sdar_steps (default B), sdar_tail (prompt tokens past the last full
  block; the client sends only the aligned prompt and strips the tail from
  the output).

If the request asks for ``logprobs=0``, the "logprob" of each emitted token
is the global denoise step that unmasked it (counted over the whole
response, as jetengine does; 0 for the prompt tail): TraceRL's step map.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from .qwen3 import Qwen3ForCausalLM


def power_step(probs, masked, thr, alpha, K=256):
    """One denoise step drawn from q*(step|z) ∝ P(step)^alpha, P = the low_confidence_dynamic
    step law at T=1 (ntr=1), plus log Z(z) = log sum_step P(step)^alpha (the SMC weight).

    P(step) has two cases (thr > 0.5, so at most the top token of a position can pass):
      A  commit set C = every position whose sample beats thr (non-empty):
         prod_{j in C} p_j(top) * prod_{k in M\\C} m_k,          m_k = mass of p_k <= thr
      F  nothing beats thr, the most confident sample (j, v) commits:
         p_j(v) * prod_{k in M, k != j} c_k(p_j(v)),             c_k(u) = mass of p_k < u
    so  Z = [prod_k (a_k + m_k^a) - prod_k m_k^a] + sum_{j, v} p_j(v)^a prod_{k != j} c_k(p_j(v))^a,
    a_k = p_k(top)^a [p_k(top) > thr]. probs [n,B,V] at T=1; masked [n,B]; thr, alpha [n].
    Returns transfer [n,B] bool, x0 [n,B] tokens, logZ [n] (0 for rows with nothing masked),
    logP [n] = log P(sampled step) under the base law, forced [n] = the step was case F.
    """
    n, B, V = probs.shape
    K = min(K, V)
    # ponytail: F only over each position's top-K tokens with p > 1/K, which makes c_k exact from the
    # top-K alone; dropped mass <= |M| K^(1-alpha) (2e-7 at K=256, alpha=4). Raise K if that matters.
    floor = 0.0 if K == V else 1.0 / K
    tv, ti = probs.topk(K, -1)                                        # descending
    al, th = alpha[:, None], thr[:, None]
    top = tv[..., 0]
    pas = masked & (top > th)
    la = torch.where(pas, al * top.log(), -math.inf)                  # log a_k
    lm = torch.where(pas, al * (1 - top).clamp_min(1e-30).log(), 0.0) # log m_k^a (1 if nothing passes)
    lam = torch.logaddexp(la, lm)
    lr = lm - lam                                                     # log P(k not committed | A-factor)
    nopass = ~pas.any(-1)
    logA = lam.sum(-1) + torch.log((-torch.expm1(lr.sum(-1))).clamp_min(1e-38))
    logA = torch.where(nopass, -math.inf, logA)
    # F: u = p_j(v) over candidates; c_k(u) = 1 - (mass of p_k >= u) from the ascending top-K cumsum
    asc = tv.flip(-1)
    cs = torch.cat([torch.zeros_like(asc[..., :1]), asc.cumsum(-1)], -1)          # exclusive cumsum
    u = tv.reshape(n, 1, B * K).expand(n, B, B * K).contiguous()                  # [n, k, (j,i)]
    idx = torch.searchsorted(asc.contiguous(), u, side="left")
    c = cs.gather(-1, idx) + (1 - cs[..., -1:]).clamp_min(0)                       # [n, k, (j,i)]
    lc = torch.where(masked[..., None], al[..., None] * c.clamp_min(1e-38).log(), 0.0).view(n, B, B, K)
    eye = torch.eye(B, dtype=torch.bool, device=probs.device)[None, :, :, None]   # k == j
    lF = (al[..., None] * tv.log() + torch.where(eye, 0.0, lc).sum(1))             # [n, j, i]
    ok = masked[..., None] & (tv <= th[..., None]) & (tv > floor)
    lF = torch.where(ok, lF, -math.inf).view(n, B * K)
    logF = lF.logsumexp(-1)
    logZ = torch.logaddexp(logA, logF)
    # sample: case A vs F, then A's commits sequentially (exactly conditioned on >= 1), or F's (j, v)
    useA = torch.rand(n, device=probs.device).log() < logA - logZ
    pi = (la - lam).exp()                                             # P(commit k) in A
    tail = lr.flip(-1).cumsum(-1).flip(-1)                            # log prod_{k' >= k} r_k'
    tA = torch.zeros_like(masked)
    none = torch.ones(n, dtype=torch.bool, device=probs.device)
    rnd = torch.rand(n, B, device=probs.device)
    for k in range(B):
        pk = torch.where(none, pi[:, k] / (-torch.expm1(tail[:, k])).clamp_min(1e-38), pi[:, k])
        tA[:, k] = pas[:, k] & (rnd[:, k] < pk)
        none &= ~tA[:, k]
    g = lF - torch.empty_like(lF).exponential_().log()                # Gumbel-max over (j, i)
    ji = g.argmax(-1)
    j, i = ji // K, ji % K
    tF = torch.zeros_like(masked)
    tF[torch.arange(n, device=probs.device), j] = True
    transfer = torch.where(useA[:, None], tA, tF)
    xF = ti[torch.arange(n, device=probs.device), j, i]
    x0 = torch.where(useA[:, None], ti[..., 0], xF[:, None].expand(n, B))
    has = masked.any(-1)
    lqA = torch.where(tA, la, torch.where(pas, lm, 0.0)).sum(-1)
    lq = torch.where(useA, lqA, lF.gather(-1, ji[:, None]).squeeze(-1))       # log P(step)^alpha
    return (transfer & has[:, None], x0, torch.where(has, logZ, 0.0),
            torch.where(has, lq / alpha, 0.0), has & ~useA)


def base_step_logp(probs, masked, thr, transfer, x0):
    """Exact log P of a realized base (T=1) low_confidence_dynamic step over the full vocab, and
    whether it was forced (case F). Same law as power_step's P; probs [n,B,V], rest as there."""
    pt = probs.gather(-1, x0[..., None]).squeeze(-1)
    tr = transfer & masked
    forced = ~(tr & (pt > thr[:, None])).any(-1)
    top = probs.amax(-1)
    lm = torch.where(top > thr[:, None], torch.log1p(-top).clamp_min(-69.0), 0.0)        # log m_k
    lA = torch.where(tr, pt.log(), torch.where(masked, lm, 0.0)).sum(-1)
    u = torch.where(tr, pt, 0.0).amax(-1)                                                  # the forced commit's p_j(v)
    c = (probs * (probs < u[:, None, None])).sum(-1)                                       # c_k(u)
    lF = u.log() + torch.where(masked & ~tr, c.clamp_min(1e-38).log(), 0.0).sum(-1)
    return torch.where(forced, lF, lA), forced & masked.any(-1)


@torch.compile(dynamic=True)
def fast_step(raw, temp, canvas, step, commit, S, thr, first, gstep, mask_id: int, B: int):
    """The default (no power / no decouple / no top-k-p) denoise step as one fused graph, no host syncs.
    Same law as the eager path: Gumbel-max on logits/T == argmax(probs / Exp(1)); p = softmax prob of the draw.
    Only the first B canvas slots are the block; any slots past B are suffix masks that are never unmasked."""
    logits = raw.float() / temp.clamp(min=1e-5)[:, None, None]
    lse = logits.logsumexp(-1)
    x0 = (logits - torch.empty_like(logits).exponential_().log()).argmax(-1)
    x0 = torch.where((temp < 1e-5)[:, None], logits.argmax(-1), x0)
    p = (logits.gather(-1, x0[..., None]).squeeze(-1) - lse).exp()
    # greedy: the T->0 law is a point mass (p == 1 everywhere, so the whole block would commit at once);
    # rank by the model's own T=1 probability instead, as LLaDA's low-confidence remasking does
    raw = raw.float()
    p1 = (raw.gather(-1, x0[..., None]).squeeze(-1) - raw.logsumexp(-1)).exp()
    p = torch.where((temp < 1e-5)[:, None], p1, p)
    block_canvas = canvas[:, :B]
    masked = block_canvas == mask_id
    conf = torch.where(masked, p, -1.0)
    ntr = B // S + (step < B % S).int()
    rank = conf.argsort(-1, descending=True).argsort(-1)
    transfer = (conf > thr[:, None]) & masked
    top = (rank < ntr[:, None]) & masked
    transfer = torch.where(transfer.sum(-1, keepdim=True) < ntr[:, None], transfer | top, transfer)
    transfer &= ~commit[:, None]
    new_block = torch.where(transfer, x0, block_canvas)
    new_canvas = torch.cat((new_block, canvas[:, B:]), dim=-1)
    new_step = step + 1
    gstep = gstep + (~commit).int()
    new_first = torch.where(transfer, gstep[:, None], first[:, :B])
    first = torch.cat((new_first, first[:, B:]), dim=-1)
    converged = ~(new_block == mask_id).any(-1) | (new_step >= S)
    c = commit[:, None]
    emit = torch.where(c, canvas, 0)
    return (emit, commit.int() * B, first,
            torch.where(c, mask_id, new_canvas), torch.where(c, 0, first), torch.where(commit, 0, new_step),
            gstep, ~commit & converged)


class SDARForBlockDiffusion(Qwen3ForCausalLM):
    @staticmethod
    def get_model_state_cls():
        return SDARModelState


class SDARModelState(DefaultModelState):
    # Decode queries are the canvas alone: no bonus token per step.
    num_new_sampled_tokens_per_step: int = 0

    def get_supported_generation_tasks(self):
        return ("generate",)

    def custom_sampler(self, sampler: Any):
        hf = self.model_config.hf_config
        return SDARSampler(
            sampler,
            block=self.vllm_config.diffusion_config.canvas_length,
            blk=getattr(hf, "block_size", self.vllm_config.diffusion_config.canvas_length),
            mask_id=hf.mask_token_id,
            max_num_reqs=self.max_num_reqs,
            device=self.device,
        ), None


class SDARSampler:
    def __init__(self, sampler, block: int, mask_id: int, max_num_reqs: int, device, blk: int | None = None):
        self.sampling_states = sampler.sampling_states
        self.logprob_token_ids_state = sampler.logprob_token_ids_state
        self.req_states = sampler.req_states
        self.penalties_state = SimpleNamespace(output_bin_counts=None)  # no penalties
        self.compute_nans = False
        self.B, self.mask_id, self.device = block, mask_id, device
        self.blk = blk or block  # commit block width; later canvas slots are context only (LLaDA)
        R = max_num_reqs
        z = dict(device=device)
        self.canvas = torch.full((R, block), mask_id, dtype=torch.int64, **z)
        self.first = torch.zeros(R, block, dtype=torch.int32, **z)  # unmask step
        self.step = torch.zeros(R, dtype=torch.int32, **z)
        self.gstep = torch.zeros(R, dtype=torch.int32, **z)  # denoise steps so far (all blocks)
        self.commit = torch.zeros(R, dtype=torch.bool, **z)
        self.temp = UvaBackedTensor(R, dtype=torch.float32)
        self.thr = UvaBackedTensor(R, dtype=torch.float32)
        self.alpha = UvaBackedTensor(R, dtype=torch.float32)  # sdar_power: >0 draws steps from q* ∝ P^alpha
        self.dec = UvaBackedTensor(R, dtype=torch.int32)      # sdar_decouple: commit test on the T=1 prob of the drawn token
        self.logw = torch.zeros(R, dtype=torch.float32, **z)   # sum of log Z over the current block
        self.logp = torch.zeros(R, dtype=torch.float32, **z)   # sum of log P(step) over the current block
        self.lfmin = torch.zeros(R, dtype=torch.float32, **z)  # min log P over the block's forced (F) steps
        self.nsteps = UvaBackedTensor(R, dtype=torch.int32)
        self._sampled = torch.zeros(R, block, dtype=torch.int32, **z)
        self._num_sampled = torch.zeros(R, dtype=torch.int32, **z)
        self._qlen = UvaBackedTensor(R, dtype=torch.int32)
        self._nlog = UvaBackedTensor(R, dtype=torch.int32)

    # -- request bookkeeping ------------------------------------------------
    def add_request(self, req_idx: int, sampling_params: Any) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)
        self.logprob_token_ids_state.add_request(req_idx, sampling_params)
        extra = getattr(sampling_params, "extra_args", None) or {}
        self.temp.np[req_idx] = float(extra.get("sdar_temperature", 1.0))
        self.thr.np[req_idx] = float(extra.get("sdar_threshold", 0.9))
        self.alpha.np[req_idx] = float(extra.get("sdar_power", 0.0))
        self.dec.np[req_idx] = int(bool(extra.get("sdar_decouple", False)))
        self.nsteps.np[req_idx] = int(extra.get("sdar_steps", self.blk))
        tail = list(extra.get("sdar_tail", ()))[: self.blk]
        seed = tail + [self.mask_id] * (self.B - len(tail))
        self.canvas[req_idx] = async_tensor_h2d(seed, dtype=torch.int64, device=self.device)
        self.first[req_idx] = 0
        self.step[req_idx] = 0
        self.gstep[req_idx] = 0
        self.commit[req_idx] = False
        self.logw[req_idx] = 0
        self.logp[req_idx] = 0
        self.lfmin[req_idx] = 0

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        self.logprob_token_ids_state.apply_staged_writes()
        self.temp.copy_to_uva()
        self.thr.copy_to_uva()
        self.alpha.copy_to_uva()
        self.dec.copy_to_uva()
        self.nsteps.copy_to_uva()

    def get_logprobs_dims(self, *args, **kwargs):  # only used with TP-sharded vocab
        raise NotImplementedError

    # -- sampling -----------------------------------------------------------
    def _finish_prefills(self, input_batch, idx_np: np.ndarray) -> None:
        done = (
            input_batch.num_computed_prefill_tokens_np[idx_np]
            + input_batch.num_scheduled_tokens[idx_np]
            >= input_batch.prefill_len_np[idx_np]
        )
        slots = input_batch.idx_mapping_np[idx_np[done]]
        if len(slots):
            s = async_tensor_h2d(slots.astype(np.int64), device=self.device)
            self.req_states.draft_tokens[s, : self.B] = self.canvas[s]

    def _output(self, input_batch, sampled, num_sampled, nlog_np, logprobs=None):
        n = input_batch.num_reqs
        self._qlen.np[:n] = np.diff(input_batch.query_start_loc_np[: n + 1])
        self._nlog.np[:n] = nlog_np
        self._qlen.copy_to_uva()
        self._nlog.copy_to_uva()
        nlog, qlen = self._nlog.gpu[:n], self._qlen.gpu[:n]
        # Denoise steps reject the whole canvas (KV is not kept); commits keep it.
        num_rejected = torch.where((nlog > 0) & (num_sampled == 0), qlen, nlog - num_sampled)
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def __call__(self, logits, input_batch, draft_logits=None) -> SamplerOutput:
        n_req, canvas_width, block_width = input_batch.num_reqs, self.B, self.blk
        slots_np = input_batch.idx_mapping_np[:n_req]
        nlog_np = np.diff(input_batch.cu_num_logits_np[: n_req + 1])
        sampled = self._sampled[:n_req]
        num_sampled = self._num_sampled[:n_req]
        sampled.zero_()
        num_sampled.zero_()

        pre_np = np.where(nlog_np == 0)[0]
        if len(pre_np):
            self._finish_prefills(input_batch, pre_np)
        dec_np = np.where(nlog_np > 0)[0]
        # No canvas rows, or a warm-up dummy batch (real canvases are always B
        # wide as long as max_model_len leaves room for the last block).
        if len(dec_np) == 0 or not (nlog_np[dec_np] == canvas_width).all():
            return self._output(input_batch, sampled, num_sampled, nlog_np)

        n = len(dec_np)
        logit_width = logits.shape[0] // n
        if logits.shape[0] != n * logit_width or logit_width not in (block_width, canvas_width):
            raise ValueError(
                f"Expected {block_width} or {canvas_width} logits per request, got {logit_width}"
            )
        raw = logits.view(n, logit_width, -1)[:, :block_width]
        slots = async_tensor_h2d(slots_np[dec_np].astype(np.int64), device=self.device)
        dec_idx = async_tensor_h2d(dec_np.astype(np.int64), device=self.device)

        top_k, top_p = self.sampling_states.get_top_k_top_p(
            slots.repeat_interleave(logit_width), slots_np[dec_np]
        )
        if logit_width != block_width:
            if top_k is not None:
                top_k = top_k.view(n, logit_width)[:, :block_width].reshape(-1)
            if top_p is not None:
                top_p = top_p.view(n, logit_width)[:, :block_width].reshape(-1)
        sl = slots_np[dec_np]
        if (top_k is None and top_p is None and not self.alpha.np[sl].any() and not self.dec.np[sl].any()
                and self.sampling_states.max_num_logprobs(slots_np) < 0):
            emit, nemit, _, canvas, first, step, gstep, commit = fast_step(
                raw, self.temp.gpu[slots], self.canvas[slots], self.step[slots], self.commit[slots],
                self.nsteps.gpu[slots], self.thr.gpu[slots], self.first[slots], self.gstep[slots], self.mask_id, block_width)
            sampled[dec_idx] = emit.to(sampled.dtype)
            num_sampled[dec_idx] = nemit
            self.canvas[slots], self.first[slots], self.step[slots] = canvas, first, step
            self.gstep[slots], self.commit[slots] = gstep, commit
            self.req_states.draft_tokens[slots, :canvas_width] = canvas
            return self._output(input_batch, sampled, num_sampled, nlog_np)

        # Temperature -> top-k -> top-p -> softmax, then one categorical draw.
        temp = self.temp.gpu[slots]
        greedy = temp < 1e-5
        logits = raw / temp.clamp(min=1e-5)[:, None, None]
        if top_k is not None or top_p is not None:
            logits = apply_top_k_top_p(logits.view(n * block_width, -1), top_k, top_p).view(n, block_width, -1)
        probs = logits.softmax(-1)
        x0 = (probs / torch.empty_like(probs).exponential_()).argmax(-1)
        x0 = torch.where(greedy[:, None], probs.argmax(-1), x0)
        p = probs.gather(-1, x0[..., None]).squeeze(-1)
        # decoupled rows: tokens come from the tempered law, but the commit test reads the model's own (T=1)
        # probability of the drawn token, so temperature cannot inflate confidences into joint parallel commits
        dec = (self.dec.gpu[slots] > 0) | greedy  # greedy: the T->0 law gives p == 1, see fast_step
        if dec.any():
            p1 = (raw.gather(-1, x0[..., None]).squeeze(-1) - raw.logsumexp(-1)).exp()
            p = torch.where(dec[:, None], p1, p)

        canvas, step, commit = self.canvas[slots], self.step[slots], self.commit[slots]
        S, thr = self.nsteps.gpu[slots], self.thr.gpu[slots]
        masked = canvas[:, :block_width] == self.mask_id
        conf = torch.where(masked, p, -1.0)
        # low_confidence_dynamic: everything above the threshold, but at least
        # ntr tokens (B // S, plus one for the first B % S steps) by confidence.
        ntr = block_width // S + (step < block_width % S).int()
        rank = conf.argsort(-1, descending=True).argsort(-1)
        transfer = (conf > thr[:, None]) & masked
        top = (rank < ntr[:, None]) & masked
        transfer = torch.where(transfer.sum(-1, keepdim=True) < ntr[:, None], transfer | top, transfer)
        alpha = self.alpha.gpu[slots]
        pw = alpha > 0
        if pw.any():  # power rows (client sends T=1, no top-k/top-p): exact q* step, weight log Z
            # alpha == 1 is the base law itself: keep the base draw, report its exact full-vocab log P
            tq, xq, lz, lp, fo = power_step(probs, masked, thr, alpha)
            one = alpha == 1
            lpb, fob = base_step_logp(probs, masked, thr, transfer, x0)
            sh = pw & ~one
            transfer = torch.where(sh[:, None], tq, transfer)
            x0 = torch.where(sh[:, None], xq, x0)
            lz, lp, fo = torch.where(one, 0.0, lz), torch.where(one, lpb, lp), torch.where(one, fob, fo)
        else:
            lz = lp = torch.zeros_like(alpha)
            fo = torch.zeros_like(pw)
        transfer &= ~commit[:, None]

        new_block = torch.where(transfer, x0, canvas[:, :block_width])
        new_canvas = torch.cat((new_block, canvas[:, block_width:]), dim=-1)
        new_step = step + 1
        gstep = self.gstep[slots] + (~commit).int()
        new_first = torch.where(transfer, gstep[:, None], self.first[slots, :block_width])
        first = torch.cat((new_first, self.first[slots, block_width:]), dim=-1)
        converged = ~(new_block == self.mask_id).any(-1) | (new_step >= S)

        # Commit rows emit their canvas, then restart from an all-mask canvas.
        sampled[dec_idx] = torch.where(commit[:, None], canvas, 0).to(sampled.dtype)
        num_sampled[dec_idx] = commit.int() * block_width
        # power rows emit [block log weight, denoise steps, block log P, min forced-step log P] in place of the step map
        logw, logp, lfmin = self.logw[slots], self.logp[slots], self.lfmin[slots]
        pinfo = F.pad(torch.stack([logw, step.float(), logp, lfmin], -1), (0, block_width - 4))
        emit_first = torch.where(pw[:, None], pinfo, first[:, :block_width].float())
        self.logw[slots] = torch.where(commit, 0.0, logw + torch.where(pw, lz, 0.0))
        live = pw & ~commit
        self.logp[slots] = torch.where(commit, 0.0, logp + torch.where(live, lp, 0.0))
        self.lfmin[slots] = torch.where(commit, 0.0, torch.where(live & fo, torch.minimum(lfmin, lp), lfmin))
        c = commit[:, None]
        self.canvas[slots] = torch.where(c, self.mask_id, new_canvas)
        self.first[slots] = torch.where(c, 0, first)
        self.step[slots] = torch.where(commit, 0, new_step)
        self.gstep[slots] = gstep
        self.commit[slots] = ~commit & converged
        self.req_states.draft_tokens[slots, :canvas_width] = self.canvas[slots]

        logprobs = None
        if self.sampling_states.max_num_logprobs(slots_np) >= 0:
            # Step map rides the logprob channel: one column, value = unmask step.
            emit_np = commit.cpu().numpy()
            if emit_np.any():
                rows = commit.nonzero().squeeze(-1)
                ids = canvas[rows, :block_width].reshape(-1, 1)
                cu, off, row_of = [], 0, dict(zip(dec_np.tolist(), emit_np.tolist()))
                for i in range(n_req):
                    cu.append(off)
                    off += block_width if row_of.get(i, False) else 0
                logprobs = LogprobsTensors(
                    logprob_token_ids=ids,
                    logprobs=emit_first[rows].reshape(-1, 1).float(),
                    selected_token_ranks=torch.zeros(ids.shape[0], dtype=torch.int64, device=self.device),
                    cu_num_generated_tokens=cu,
                )
        return self._output(input_batch, sampled, num_sampled, nlog_np, logprobs)


# LLaDA-8B (bidirectional MDM) served the same way: block-causal canvas of B = 32 with a commit pass.
# This is Fast-dLLM's prefix cache without the per-block refresh (an approximation for LLaDA, exact for SDAR).
from .llama import LlamaForCausalLM  # noqa: E402

_LLADA_NAMES = [("model.transformer.wte.", "model.embed_tokens."), ("model.transformer.ln_f.", "model.norm."),
                ("model.transformer.ff_out.", "lm_head."), ("model.transformer.blocks.", "model.layers."),
                (".attn_out.", ".self_attn.o_proj."), (".q_proj.", ".self_attn.q_proj."), (".k_proj.", ".self_attn.k_proj."),
                (".v_proj.", ".self_attn.v_proj."), (".ff_proj.", ".mlp.gate_proj."), (".up_proj.", ".mlp.up_proj."),
                (".ff_out.", ".mlp.down_proj."), (".attn_norm.", ".input_layernorm."), (".ff_norm.", ".post_attention_layernorm.")]


def _llada_name(n: str) -> str:
    for a, b in _LLADA_NAMES:
        n = n.replace(a, b)
    return n


class LLaDAForBlockDiffusion(LlamaForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @staticmethod
    def get_model_state_cls():
        return SDARModelState

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return super().compute_logits(hidden_states)

    def load_weights(self, weights):
        return super().load_weights((_llada_name(n), w) for n, w in weights)
