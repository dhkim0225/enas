import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.embed = nn.Embedding(6, 64)
        self.lstm = nn.LSTMCell(64, 64)
        self.w_soft = nn.Linear(64, 5, bias=False)
        self.b_soft = nn.Parameter([[10, 10, 0, 0, 0]])
        self.b_soft_no_learn = torch.Tensor([[0.25, 0.25, -0.25, -0.25, -0.25]]).requires_grad_(False).cuda()

        # attention
        self.w_attn_1 = nn.Linear(64, 64, bias=False)
        self.w_attn_2 = nn.Linear(64, 64, bias=False)
        self.v_attn = nn.Linear(64, 1, bias=False)
        self.reset_param()

    def reset_param(self):
        for name, param in self.named_parameters():
            if 'b_soft' not in name:
                nn.init.uniform_(param, -0.1, 0.1)

    def forward(self):
        arc_seq_1, entropy_1, log_prob_1, c, h = self.sampler(use_bias=True)
        arc_seq_2, entropy_2, log_prob_2, _, _ = self.sampler(prev_c=c, prev_h=h)

        sample_arc = (arc_seq_1, arc_seq_2)
        sample_log_prob = log_prob_1 + log_prob_2
        sample_entropy = entropy_1 + entropy_2

        return sample_arc, sample_log_prob, sample_entropy

    def sampler(self, prev_c=None, prev_h=None, use_bias=False):
        if prev_c is None:
            prev_c = torch.zeros(1, 64).cuda()
            prev_h = torch.zeros(1, 64).cuda()

        anchors = list()
        anchors_w_1 = list()
        arc_seq = list()

        inputs = self.embed(torch.zeros(1).long().cuda())
        for layer_id in range(2):
            embed = inputs
            next_h, next_c = self.lstm(embed, (prev_h, prev_c))
            prev_h, prev_c = next_h, next_c
            anchors.append(torch.zeros(next_h.shape).cuda())
            anchors_w_1.append(self.w_attn_1(next_h))

        entropy = list()
        log_prob = list()

        for layer_id in range(2, 7):
            prev_layers = []
            for i in range(2):  # index_1, index_2
                embed = inputs
                next_h, next_c = self.lstm(embed, (prev_h, prev_c))
                prev_h, prev_c = next_h, next_c
                query = torch.stack(anchors_w_1[:layer_id], dim=1)
                query = query.view(layer_id, self.lstm_size)
                query = torch.tanh(query + self.w_attn_2(next_h))
                query = self.v_attn(query)
                logits = query.view(1, layer_id)
                logits = logits/5.0 + 1.10 * torch.tanh(logits)

                prob = F.softmax(logits, dim=-1)
                index = torch.multinomial(prob, 1).long().view(1)
                arc_seq.append(index)
                arc_seq.append(0)
                log_prob.append(F.cross_entropy(logits, index))
                curr_ent = -torch.mean(torch.sum(torch.mul(F.log_softmax(logits, dim=-1), prob), dim=1)).detach()

                entropy.append(curr_ent)
                prev_layers.append(anchors[index])
                inputs = prev_layers[-1].view(1, -1).requires_grad_()

            for i in range(2):  # op_1, op_2
                embed = inputs
                next_h, next_c = self.lstm(embed, (prev_h, prev_c))
                prev_c, prev_h = next_c, next_h

                logits = self.w_soft(next_h) + self.b_soft.requires_grad_()
                logits = (1.10 / 2.5) * torch.tanh(logits / 5.0)
                if use_bias:
                    logits += self.b_soft_no_learn
                prob = F.softmax(logits, dim=-1)

                op_id = torch.multinomial(prob, 1).long().view(1)
                arc_seq[2*i - 3] = op_id
                log_prob.append(F.cross_entropy(logits, op_id))
                curr_ent = -torch.mean(torch.sum(torch.mul(F.log_softmax(logits, dim=-1), prob), dim=1)).detach()
                entropy.append(curr_ent)
                inputs = self.embed(op_id + 1)

            next_h, next_c = self.lstm(inputs, (prev_h, prev_c))
            prev_h, prev_c = next_h, next_c
            anchors.append(next_h)
            anchors_w_1.append(self.w_attn_1(next_h))
            inputs = self.embed(torch.zeros(1).long().cuda())

        arc_seq = torch.tensor(arc_seq)
        entropy = sum(entropy)
        log_prob = sum(log_prob)
        last_h, last_c = next_h, next_c

        return arc_seq, entropy, log_prob, last_c, last_h

