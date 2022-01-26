import torch


def build_mask_type1(utt_mask, spk_mask=None, bidirectional=False):
    utt_mask = torch.matmul(utt_mask.unsqueeze(2), utt_mask.unsqueeze(1))
    if bidirectional is False:
        utt_mask = utt_mask.tril(0)
    umask = utt_mask.eq(0)
    if spk_mask is not None:
        batch_size = spk_mask.size(0)
        seq_len = spk_mask.size(1)
        mask1 = spk_mask.unsqueeze(2).expand(batch_size, seq_len, seq_len)
        mask2 = spk_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        smask = torch.eq(mask1, mask2)
        smask = torch.masked_fill(smask, umask, False)
        smask = torch.eq(smask, False)
        return umask, smask
    return umask, None


def build_mixed_mask_prior(utt_mask, spk_mask=None, bidirectional=False):
    # utt_mask: (bsz, slen)
    utt_mask = torch.matmul(utt_mask.unsqueeze(2), utt_mask.unsqueeze(1))
    if bidirectional is False:
        utt_mask = utt_mask.tril(0)
    umask = utt_mask.eq(0)
    if spk_mask is not None:
        batch_size = spk_mask.size(0)
        seq_len = spk_mask.size(1)
        mask1 = spk_mask.unsqueeze(2).expand(batch_size, seq_len, seq_len)
        mask2 = spk_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        smask_self = torch.eq(mask1, mask2)
        smask_other = torch.eq(smask_self, False)
        smask_self = torch.masked_fill(smask_self, umask, False)
        smask_other = torch.masked_fill(smask_other, umask, False)

        smask_self = torch.eq(smask_self, False)
        smask_other = torch.eq(smask_other, False)
        return umask, smask_self, smask_other
    return umask, None, None


def build_mixed_mask_local(utt_mask, spk_mask=None, window=10, bidirectional=False):
    # utt_mask: (bsz, slen)
    utt_mask = torch.matmul(utt_mask.unsqueeze(2), utt_mask.unsqueeze(1))
    utt_mask = utt_mask.tril(window)-utt_mask.tril(-window-1)
    if bidirectional is False:
        utt_mask = utt_mask.tril(0)
    umask = utt_mask.eq(0)
    if spk_mask is not None:
        batch_size = spk_mask.size(0)
        seq_len = spk_mask.size(1)
        mask1 = spk_mask.unsqueeze(2).expand(batch_size, seq_len, seq_len)
        mask2 = spk_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        smask_self = torch.eq(mask1, mask2)
        smask_other = torch.eq(smask_self, False)
        smask_self = torch.masked_fill(smask_self, umask, False)
        smask_other = torch.masked_fill(smask_other, umask, False)

        smask_self = torch.eq(smask_self, False)
        smask_other = torch.eq(smask_other, False)
        return umask, smask_self, smask_other
    return umask, None, None


def build_mixed_mask_post(utt_mask, spk_mask):
    utt_mask = torch.matmul(utt_mask.unsqueeze(2), utt_mask.unsqueeze(1))
    umask = utt_mask.triu(0)
    umask = umask.eq(0)
    if spk_mask is not None:
        batch_size = spk_mask.size(0)
        seq_len = spk_mask.size(1)
        mask1 = spk_mask.unsqueeze(2).expand(batch_size, seq_len, seq_len)
        mask2 = spk_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        smask_self = torch.eq(mask1, mask2)
        smask_other = torch.eq(smask_self, False)

        smask_self = torch.masked_fill(smask_self, umask, False)
        smask_other = torch.masked_fill(smask_other, umask, False)

        smask_self = torch.eq(smask_self, False)
        smask_other = torch.eq(smask_other, False)
        return umask, smask_self, smask_other
    return umask, None, None
