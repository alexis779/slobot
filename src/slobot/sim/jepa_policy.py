from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Deque, List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from slobot.sim.golf_ball_env import GolfBallEnv
from slobot.sim.gym.types import GolfBallEnvAction, GolfBallEnvObservation

from src.jepa_predictor import ActionConditionedPredictor, encode_video
from src.models.vision_transformer import vit_large


@dataclass
class _JepaModelConfig:
    encoder_embed_dim: int = 1024
    encoder_num_patches: int = 196
    target_num_patches: int = 196
    action_dim: int = 6
    pred_depth: int = 4
    pred_heads: int = 8
    pred_dim: int = 384
    dropout: float = 0.1
    context_frames: int = 6
    target_frames: int = 2
    img_size: int = 256


def _merge_jepa_cfg(saved: Optional[dict]) -> _JepaModelConfig:
    base = _JepaModelConfig()
    if not saved:
        return base
    kwargs = {
        k: saved[k]
        for k in _JepaModelConfig.__dataclass_fields__
        if k in saved
    }
    return replace(base, **kwargs)


def _load_predictor_state(path: Path) -> tuple[dict, _JepaModelConfig]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"], _merge_jepa_cfg(blob.get("config"))
    return blob, _JepaModelConfig()


class JepaPolicy:
    """
    Runs the SO-100 JEPA action-conditioned predictor with a frozen ViT-L encoder.

    The checkpoint predicts future visual latents from past actions; control targets
    are recovered by optimizing joint commands so that those latents match a short
    target clip (here: repeating the latest camera frame), then clipping to DOF limits.
    """

    def __init__(
        self,
        golf_ball_env: GolfBallEnv,
        encoder_ckpt: Union[str, Path],
        predictor_ckpt: Union[str, Path],
        *,
        camera: Literal["side", "link"] = "side",
        device: Optional[torch.device] = None,
        optimize_steps: int = 40,
        optimize_lr: float = 0.08,
    ) -> None:
        self._env = golf_ball_env
        self._camera_key = (
            "side_camera_image" if camera == "side" else "link_camera_image"
        )
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._optimize_steps = optimize_steps
        self._optimize_lr = optimize_lr

        encoder_ckpt = Path(encoder_ckpt)
        predictor_ckpt = Path(predictor_ckpt)

        state_dict, cfg = _load_predictor_state(predictor_ckpt)
        self._cfg = cfg

        self._frame_buf: Deque[torch.Tensor] = deque(maxlen=cfg.context_frames)

        self._img_tf = transforms.Compose(
            [
                transforms.Resize(
                    (cfg.img_size, cfg.img_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.encoder = self._build_encoder(encoder_ckpt, cfg)
        self.predictor = ActionConditionedPredictor(cfg).to(self._device)
        self.predictor.load_state_dict(state_dict)
        self.encoder.eval()
        self.predictor.eval()

        lo, hi = golf_ball_env.arm.genesis.entity.get_dofs_limit()
        self._q_lo = lo.to(dtype=torch.float32, device=self._device)
        self._q_hi = hi.to(dtype=torch.float32, device=self._device)

    def _build_encoder(self, encoder_ckpt: Path, cfg: _JepaModelConfig) -> nn.Module:
        enc = vit_large(
            patch_size=16,
            img_size=cfg.img_size,
            num_frames=cfg.context_frames,
            tubelet_size=2,
            uniform_power=False,
            use_sdpa=True,
        )
        raw = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
        enc_state = raw["encoder"]
        enc_state = {k.replace("module.", ""): v for k, v in enc_state.items()}
        enc.load_state_dict(enc_state, strict=False)
        enc = enc.to(self._device)
        for p in enc.parameters():
            p.requires_grad = False
        return enc

    def reset(self) -> None:
        self._frame_buf.clear()

    def _preprocess_frame(self, obs: GolfBallEnvObservation) -> torch.Tensor:
        rgb = np.asarray(obs[self._camera_key], dtype=np.uint8)
        pil = Image.fromarray(rgb)
        return self._img_tf(pil)

    def _padded_context(self) -> List[torch.Tensor]:
        buf = list(self._frame_buf)
        if not buf:
            raise RuntimeError("JepaPolicy: empty frame buffer; call act(obs) first.")
        while len(buf) < self._cfg.context_frames:
            buf = [buf[0]] + buf
        return buf[-self._cfg.context_frames :]

    def act(self, obs: GolfBallEnvObservation) -> GolfBallEnvAction:
        frame = self._preprocess_frame(obs)
        self._frame_buf.append(frame)

        ctx_list = self._padded_context()
        ctx_video = torch.stack(ctx_list, dim=0).unsqueeze(0).to(self._device)
        last = ctx_list[-1]
        tgt_video = (
            torch.stack([last, last], dim=0).unsqueeze(0).to(self._device)
        )

        with torch.no_grad():
            z_ctx = encode_video(self.encoder, ctx_video, self._device)
            z_tgt = encode_video(self.encoder, tgt_video, self._device)

        q0 = torch.as_tensor(obs["qpos"], dtype=torch.float32, device=self._device)
        actions = (
            q0.unsqueeze(0)
            .unsqueeze(0)
            .expand(1, self._cfg.context_frames, self._cfg.action_dim)
            .contiguous()
            .requires_grad_(True)
        )
        opt = torch.optim.Adam([actions], lr=self._optimize_lr)
        use_amp = self._device.type == "cuda"

        for _ in range(self._optimize_steps):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(self._device.type, enabled=use_amp):
                pred = self.predictor(z_ctx, actions)
                loss = F.l1_loss(pred, z_tgt)
            loss.backward()
            opt.step()

        cmd = actions[0, -1].detach()
        cmd = torch.maximum(torch.minimum(cmd, self._q_hi), self._q_lo)
        control = cmd.cpu().numpy().astype(np.float32)
        return GolfBallEnvAction(control_qpos=control)
