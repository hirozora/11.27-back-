"""Matched MLP with an optional candidate-to-visible-event ownership bit."""
import torch
from torch import nn
from torch.nn import functional as F


def ownership_mask(token, player, window_actor):
    return (
        (player[:, :, None] == window_actor[:, None, :])
        & (player[:, :, None] > 0)
        & (token[:, None, :] > 0)
    )


class CandidateMLP(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.event = nn.Embedding(vocab_size, 8, padding_idx=0)
        self.zone_emb = nn.Embedding(10, 4)
        self.sequence = nn.Sequential(
            nn.Linear(32 * 13 + 3, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(128, 64), nn.GELU(),
        )
        # Same size and initialization in both arms; relation-off supplies zeros.
        self.candidate = nn.Sequential(nn.Linear(64 + 22 + 8 + 32, 64), nn.GELU(), nn.Dropout(0.1))
        self.actor = nn.Linear(64, 1)
        self.location = nn.Linear(64, 9)
        self.action = nn.Linear(64, 10)

    def forward(self, x, relation, head):
        tok = x["token"].long()
        ctx = torch.cat((self.event(tok), self.zone_emb(x["wzone"].long()), x["age"].float().unsqueeze(-1)), -1)
        ctx = self.sequence(torch.cat(((ctx * tok.gt(0).unsqueeze(-1)).flatten(1), x["context"].float()), -1))
        player = x["player"]
        valid = player > 0
        owns = ownership_mask(tok, player, x["window_actor"])
        if not relation:
            owns = torch.zeros_like(owns)
        group = F.one_hot(x["group"].long(), 8).float()
        state = self.candidate(torch.cat((ctx[:, None].expand(-1, player.size(1), -1),
                                          x["hist"].float(), group, owns.float()), -1))
        q = self.actor(state).squeeze(-1).masked_fill(~valid, -torch.inf).log_softmax(-1)
        z = self.location(state).log_softmax(-1)
        if head == "context":
            r = self.action(ctx).log_softmax(-1)[:, None].expand(-1, player.size(1), -1)
        elif head == "mixture":
            r = self.action(state).log_softmax(-1)
        else:
            raise ValueError(head)
        return q, z, r


def marginal(log_q, log_r):
    return torch.logsumexp(log_q.unsqueeze(-1) + log_r, dim=1)


def role_roll(player, groups, history):
    """Permute event distributions only among same-role valid candidates."""
    valid = player > 0
    slots = torch.arange(player.size(1), device=player.device)
    same = (groups[:, :, None] == groups[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    same = same & (groups[:, :, None] < 7)
    others = same & (slots[None, None, :] < slots[None, :, None])
    prev = torch.where(others, slots, -1).amax(-1)
    wrap = torch.where(same, slots, -1).amax(-1)
    pick = torch.where(prev >= 0, prev, wrap)
    pick = torch.where(valid & (groups < 7), pick, slots)
    return history.gather(1, pick[..., None].expand_as(history)), (same.sum(-1) > 1).any(-1)
