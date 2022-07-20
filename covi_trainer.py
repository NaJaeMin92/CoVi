import torch
from utils import *


def train_covi(args, src_train_loader, tgt_train_loader, optimizer, optimizer_emp, model, net_part1, emp_learner, entropy, cross_entropy, epoch):
    print("Epoch: [{}/{}]".format(epoch, args.epochs))
    set_model_mode('train', [model, net_part1, emp_learner])
    for step, (src_data, tgt_data) in enumerate(zip(src_train_loader, tgt_train_loader)):
        src_imgs, src_labels = src_data
        tgt_imgs, tgt_labels = tgt_data
        src_imgs, src_labels = src_imgs.cuda(non_blocking=True), src_labels.cuda(non_blocking=True)
        tgt_imgs = tgt_imgs.cuda(non_blocking=True)

        t_out = model(tgt_imgs)
        top1_label, top2_label, top1_prob = get_top2(t_out)
        prob_mean, prob_std = top1_prob.mean(), top1_prob.std()

        # Train emp_learner
        s_out_, t_out_ = net_part1(src_imgs), net_part1(tgt_imgs)
        concated = torch.cat([s_out_, t_out_], dim=1)

        emp = emp_learner(concated)
        emp, _, _ = get_top2(emp)
        emp = emp.to(torch.float32) * 0.1
        emp = torch.clamp(emp, min=args.cmin, max=args.cmax)
        vicinal_instance = get_vicinal_instance(src_imgs, tgt_imgs, emp, args.batch_size)
        vicinal_out = model(vicinal_instance)

        emp_loss = entropy(vicinal_out)

        optimizer.zero_grad()
        optimizer_emp.zero_grad()
        emp_loss.backward()
        optimizer_emp.step()

        vicinal_instance = get_vicinal_instance(src_imgs, tgt_imgs, emp.detach(), args.batch_size)
        vicinal_out = model(vicinal_instance)

        total_loss, emp_mixup_loss = 0, 0
        for i in range(args.batch_size):
            emp_mixup_loss += sample_wise_loss(vicinal_out[i], src_labels[i], top1_label[i], emp[i].detach())
        emp_mixup_loss = emp_mixup_loss / args.batch_size
        total_loss = emp_mixup_loss

        # Compute contrastive loss
        with torch.no_grad():
            assert args.swap_upper > args.swap_lower

            upper_ratio = emp.detach() + args.swap_margin
            lower_ratio = emp.detach() - args.swap_margin

            upper_mask = upper_ratio.le(args.swap_upper)
            lower_mask = lower_ratio.ge(args.swap_lower)
            threshold = prob_mean - args.swap_th * prob_std
            th_mask = top1_prob.ge(threshold)
            mask_idx = torch.nonzero(th_mask & upper_mask & lower_mask).squeeze()
            upper_ratio = upper_ratio[mask_idx]
            lower_ratio = lower_ratio[mask_idx]

        if mask_idx.dim() > 0 and torch.numel(mask_idx) > 0:
            num_of_mask = len(mask_idx)
            upper_instance = get_vicinal_instance(src_imgs[mask_idx], tgt_imgs[mask_idx], upper_ratio, num_of_mask)
            lower_instance = get_vicinal_instance(src_imgs[mask_idx], tgt_imgs[mask_idx], lower_ratio, num_of_mask)
            upper_out, lower_out = model(upper_instance), model(lower_instance)
            top1_upper, top2_upper, prob_upper = get_top2(upper_out)
            top1_lower, top2_lower, prob_lower = get_top2(lower_out)

            swap_src_labels = src_labels[mask_idx]
            pure_tgt_top1 = top1_label[mask_idx]
            swap_ul_loss, swap_lu_loss = 0, 0
            for i in range(num_of_mask):
                swap_ul_loss += sample_wise_loss(upper_out[i], swap_src_labels[i], top1_lower[i], upper_ratio[i])
                swap_lu_loss += sample_wise_loss(lower_out[i], pure_tgt_top1[i], top1_upper[i].cuda(), lower_ratio[i])

            swap_ul_loss = swap_ul_loss / num_of_mask / 2
            swap_lu_loss = swap_lu_loss / num_of_mask / 2
            total_loss += swap_ul_loss
            total_loss += swap_lu_loss

        # Compute consensus loss
        shuff_idx = torch.randperm(args.batch_size).cuda(non_blocking=True)
        i = 1 - args.consensus_ratio
        mixed_input = src_imgs * i + tgt_imgs * (1 - i)
        shuff_input = src_imgs * i + tgt_imgs[shuff_idx] * (1 - i)

        shuff_out1, shuff_out2 = model(mixed_input), model(shuff_input)

        consensus_loss = cross_entropy(shuff_out1, src_labels) + cross_entropy(shuff_out2, src_labels)
        total_loss += (consensus_loss / 2)

        optimizer.zero_grad()
        optimizer_emp.zero_grad()
        total_loss.backward()
        optimizer.step()
