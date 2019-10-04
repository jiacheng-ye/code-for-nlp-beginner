import torch
from torch import nn


class CRF(nn.Module):
    def __init__(self, label_size):
        '''label_size = real size + 2, included START and END '''
        super(CRF, self).__init__()

        self.label_size = label_size
        self.start = self.label_size - 2
        self.end = self.label_size - 1
        transition = torch.randn(self.label_size, self.label_size)
        self.transition = nn.Parameter(transition)
        self.initialize()

    def initialize(self):
        nn.init.uniform_(self.transition.data, -0.1, 0.1)
        self.transition.data[:, self.end] = -1000.0
        self.transition.data[self.start, :] = -1000.0

    def pad_logits(self, logits):
        batch_size, seq_len, label_num = logits.size()
        pads = logits.new_full((batch_size, seq_len, 2), -1000.0,
                               requires_grad=False)
        logits = torch.cat([logits, pads], dim=2)
        return logits

    def calc_binary_score(self, labels, predict_mask):
        '''
        Gold transition score
        :param labels: [batch_size, seq_len] LongTensor
        :param predict_mask: [batch_size, seq_len] LongTensor
        :return: [batch_size] FloatTensor
        '''
        batch_size, seq_len = labels.size()

        labels_ext = labels.new_empty((batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start
        labels_ext[:, 1:-1] = labels
        labels_ext[:, -1] = self.end
        pad = predict_mask.new_ones([batch_size, 1], requires_grad=False)
        pad_stop = labels.new_full([batch_size, 1], self.end, requires_grad=False)
        mask = torch.cat([pad, predict_mask, pad], dim=-1).long()
        labels = (1 - mask) * pad_stop + mask * labels_ext

        trn = self.transition
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = torch.cat([pad, predict_mask], dim=-1).float()
        trn_scr = trn_scr * mask
        score = trn_scr

        return score

    def calc_unary_score(self, logits, labels, predict_mask):
        '''
        Gold logits score
        :param logits: [batch_size, seq_len, n_labels] FloatTensor
        :param labels: [batch_size, seq_len] LongTensor
        :param predict_mask: [batch_size, seq_len] LongTensor
        :return: [batch_size] FloatTensor
        '''
        labels_exp = labels.unsqueeze(-1)
        scores = torch.gather(logits, 2, labels_exp).squeeze(-1)
        scores = scores * predict_mask.float()
        return scores

    def calc_gold_score(self, logits, labels, predict_mask):
        '''
        Total score of gold sequence.
        :param logits: [batch_size, seq_len, n_labels] FloatTensor
        :param labels: [batch_size, seq_len] LongTensor
        :param predict_mask: [batch_size, seq_len] LongTensor
        :return: [batch_size] FloatTensor
        '''
        unary_score = self.calc_unary_score(logits, labels, predict_mask).sum(
            1).squeeze(-1)
        # print(unary_score)
        binary_score = self.calc_binary_score(labels, predict_mask).sum(1).squeeze(-1)
        # print(binary_score)
        return unary_score + binary_score

    def calc_norm_score(self, logits, predict_mask):
        '''
        Total score of all sequences.
        :param logits: [batch_size, seq_len, n_labels] FloatTensor
        :param predict_mask: [batch_size, seq_len] LongTensor
        :return: [batch_size] FloatTensor
        '''
        batch_size, seq_len, feat_dim = logits.size()

        alpha = logits.new_full((batch_size, self.label_size), -100.0)
        alpha[:, self.start] = 0

        predict_mask_ = predict_mask.clone()  # (batch_size, max_seq)

        logits_t = logits.transpose(1, 0)  # (max_seq, batch_size, num_labels + 2)
        predict_mask_ = predict_mask_.transpose(1, 0)  # (max_seq, batch_size)
        for word_mask_, logit in zip(predict_mask_, logits_t):
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   *self.transition.size())
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  *self.transition.size())
            trans_exp = self.transition.unsqueeze(0).expand_as(alpha_exp)
            mat = logit_exp + alpha_exp + trans_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = word_mask_.float().unsqueeze(-1).expand_as(alpha)  # (batch_size, num_labels+2)
            alpha = mask * alpha_nxt + (1 - mask) * alpha

        alpha = alpha + self.transition[self.end].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def viterbi_decode(self, logits, predict_mask):
        """
        :param logits: [batch_size, seq_len, n_labels] FloatTensor
        :param predict_mask: [batch_size, seq_len] LongTensor
        :return scores: [batch_size] FloatTensor
        :return paths: [batch_size, seq_len] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        vit = logits.new_full((batch_size, self.label_size), -100.0)
        vit[:, self.start] = 0
        predict_mask_ = predict_mask.clone()  # (batch_size, max_seq)
        predict_mask_ = predict_mask_.transpose(1, 0)  # (max_seq, batch_size)
        logits_t = logits.transpose(1, 0)
        pointers = []
        for ix, logit in enumerate(logits_t):
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transition.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = predict_mask_[ix].float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (predict_mask_[ix:].sum(0) == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transition[self.end].unsqueeze(
                0).expand_as(vit_nxt)

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths


def log_sum_exp(tensor, dim=0):
    """LogSumExp operation."""
    m, _ = torch.max(tensor, dim)
    m_exp = m.unsqueeze(-1).expand_as(tensor)
    return m + torch.log(torch.sum(torch.exp(tensor - m_exp), dim))


def test():
    torch.manual_seed(2)
    logits = torch.tensor([[[1.2, 2.1], [2.8, 2.1], [2.2, -2.1]], [[4.1, 2.2], [2.8, 2.1], [2.2, -2.1]]])  # 2, 3, 2
    predict_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])  # 2, 3
    labels = torch.tensor([[1, 0, 0], [0, 1, 1]])  # 2, 3

    crf = CRF(4)
    logits = crf.pad_logits(logits)
    norm_score = crf.calc_norm_score(logits, predict_mask)
    print(norm_score)
    gold_score = crf.calc_gold_score(logits, labels, predict_mask)
    print(gold_score)
    loglik = gold_score - norm_score
    print(loglik)
    print(crf.viterbi_decode(logits, predict_mask))

# test()
