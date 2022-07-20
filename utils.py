import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder

from network import models


def get_data_info(args):
    if args.target in ['amazon', 'dslr', 'webcam']:  # Office-31
        resnet_type = 50
        num_classes = 31
    else:  # Office-Home
        resnet_type = 50
        num_classes = 65

    return num_classes, resnet_type


def get_dataset(domain_name, db_path):
    if domain_name in ['amazon', 'dslr', 'webcam']:  # OFFICE-31
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        tr_dataset = ImageFolder(db_path + '/office31/' + domain_name + '/', data_transforms['train'])
        te_dataset = ImageFolder(db_path + '/office31/' + domain_name + '/', data_transforms['test'])

    elif domain_name in ['art', 'product', 'clipart', 'realworld']:  # Office-Home
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        tr_dataset = ImageFolder(db_path + '/OfficeHome/OfficeHomeDataset/' + domain_name + '/', data_transforms['train'])
        te_dataset = ImageFolder(db_path + '/OfficeHome/OfficeHomeDataset/' + domain_name + '/', data_transforms['test'])

    else:
        raise ValueError('Domain %s is not found!' % domain_name)

    print('{} train set size: {}'.format(domain_name, len(tr_dataset)))
    print('{} test set size: {}'.format(domain_name, len(te_dataset)))
    return tr_dataset, te_dataset


def get_train_info():
    lr = 0.002
    l2_decay = 5e-4
    momentum = 0.9

    print('lr, l2_decay, momentum:', lr, l2_decay, momentum)
    return lr, l2_decay, momentum


def get_net_info(num_classes):
    net = nn.DataParallel(models.ResNet50().encoder).cuda()
    head = nn.DataParallel(models.Head()).cuda()
    classifier = nn.DataParallel(nn.Linear(256, num_classes)).cuda()
    emp_learner = nn.DataParallel(models.EmpLearner()).cuda()

    return net, head, classifier, emp_learner


def load_net(args, net, head, classifier, root):
    print("Load pre-trained model !")
    save_folder = root + args.baseline_path
    net.module.load_state_dict(torch.load(save_folder + '/net.pt'), strict=False)
    head.module.load_state_dict(torch.load(save_folder + '/head.pt'), strict=False)
    classifier.module.load_state_dict(torch.load(save_folder + '/classifier.pt'), strict=False)
    return net, head, classifier


def set_model_mode(mode, models):
    for model in models:
        if mode == 'train':
            model.train()
        else:
            model.eval()


def sample_wise_loss(pred, y_a, y_b, lam):
    pred = F.log_softmax(pred.unsqueeze(dim=0), dim=-1)
    return lam * F.nll_loss(pred, y_a.unsqueeze(dim=0)).cuda() + (1 - lam) * F.nll_loss(pred, y_b.unsqueeze(dim=0)).cuda()


def get_top2(q):
    topk_prob, topk_label = torch.topk(F.softmax(q, dim=1), k=2)
    topk_label = topk_label.squeeze()
    top1_label, top2_label = topk_label.t()[0], topk_label.t()[1]
    top1_prob = topk_prob.squeeze().t()[0]
    return top1_label.detach(), top2_label.detach(), top1_prob.detach()


def get_vicinal_instance(src_imgs, tgt_imgs, emp, num_of_samples):
    mixed_input = []
    for i in range(num_of_samples):
        mixed_input.append(src_imgs[i] * emp[i] + tgt_imgs[i] * (1 - emp[i]))

    mixed_input = torch.stack(mixed_input)
    return mixed_input


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        out = F.softmax(x, dim=1)
        loss = torch.mean(torch.sum(out * (torch.log(out + 1e-5)), 1))
        return loss


def evaluate(model, loader):
    total, correct = 0, 0
    set_model_mode('eval', model)
    with torch.no_grad():
        for step, tgt_data in enumerate(loader):
            tgt_imgs, tgt_labels = tgt_data
            tgt_imgs, tgt_labels = tgt_imgs.cuda(non_blocking=True), tgt_labels.cuda(non_blocking=True)
            tgt_pred = model(tgt_imgs)
            pred = tgt_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(tgt_labels.long().view_as(pred)).sum().item()
            total += tgt_labels.size(0)
    acc = (correct / total) * 100
    print('Accuracy: {:.2f}%'.format(acc))
    set_model_mode('train', model)

    return acc
