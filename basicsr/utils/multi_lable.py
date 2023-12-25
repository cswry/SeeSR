import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
# import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
import math
import numpy as np
# from util import *

# tqdm.monitor_interval = 0
# class Engine(object):
#     def __init__(self, state={}):
#         self.state = state
#         if self._state('use_gpu') is None:
#             self.state['use_gpu'] = torch.cuda.is_available()

#         if self._state('image_size') is None:
#             self.state['image_size'] = 224

#         if self._state('batch_size') is None:
#             self.state['batch_size'] = 64

#         if self._state('workers') is None:
#             self.state['workers'] = 25

#         if self._state('device_ids') is None:
#             self.state['device_ids'] = None

#         if self._state('evaluate') is None:
#             self.state['evaluate'] = False

#         if self._state('start_epoch') is None:
#             self.state['start_epoch'] = 0

#         if self._state('max_epochs') is None:
#             self.state['max_epochs'] = 90

#         if self._state('epoch_step') is None:
#             self.state['epoch_step'] = []

#         # meters
#         self.state['meter_loss'] = tnt.meter.AverageValueMeter()
#         # time measure
#         self.state['batch_time'] = tnt.meter.AverageValueMeter()
#         self.state['data_time'] = tnt.meter.AverageValueMeter()
#         # display parameters
#         if self._state('use_pb') is None:
#             self.state['use_pb'] = False
#         if self._state('print_freq') is None:
#             self.state['print_freq'] = 10
#         if not self.state['evaluate']:
#             self.state['log_path'] = open(self.state['save_model_path']+'train.log', 'a')
#         else:
#             self.state['log_path'] = open(self.state['save_model_path']+'eval.log', 'a')

#     def _state(self, name):
#         if name in self.state:
#             return self.state[name]

#     def init_learning(self, model, criterion):

#         if self._state('train_transform') is None:
#             normalize = transforms.Normalize(mean=model.image_normalization_mean,
#                                              std=model.image_normalization_std)
#             #flip = transforms.RandomChoice([transforms.RandomHorizontalFlip(),\
#                                                         #transforms.RandomVerticalFlip()])
#             self.state['train_transform'] = transforms.Compose([
#                 transforms.Resize((512, 512)),
#                 MultiScaleCrop(self.state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 normalize,
#                 transforms.RandomErasing(p=0.7),
#             ])
#             print(self.state['train_transform'])

#         if self._state('val_transform') is None:
#             normalize = transforms.Normalize(mean=model.image_normalization_mean,
#                                              std=model.image_normalization_std)
#             self.state['val_transform'] = transforms.Compose([
#                 Warp(self.state['image_size']),
#                 transforms.ToTensor(),
#                 normalize,
#             ])

#         self.state['best_score'] = 0

#     def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):

#         self.init_learning(model, criterion)

#         # define train and val transform
#         train_dataset.transform = self.state['train_transform']
#         train_dataset.target_transform = self._state('train_target_transform')
#         val_dataset.transform = self.state['val_transform']
#         val_dataset.target_transform = self._state('val_target_transform')

#         # data loading code
#         train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                    batch_size=self.state['batch_size'], shuffle=True,
#                                                    num_workers=self.state['workers'],drop_last=True)

#         val_loader = torch.utils.data.DataLoader(val_dataset,
#                                                  batch_size=self.state['batch_size'], shuffle=False,
#                                                  num_workers=self.state['workers'])

#         # optionally resume from a checkpoint
#         if self._state('resume') is not None:
#             if os.path.isfile(self.state['resume']):
#                 print("=> loading checkpoint '{}'".format(self.state['resume']))
#                 checkpoint = torch.load(self.state['resume'])
#                 self.state['start_epoch'] = checkpoint['epoch']
#                 self.state['best_score'] = checkpoint['best_score']
#                 model.load_state_dict(checkpoint['state_dict'])
#                 print("=> loaded checkpoint '{}' (epoch {})"
#                       .format(self.state['evaluate'], checkpoint['epoch']))
#             else:
#                 print("=> no checkpoint found at '{}'".format(self.state['resume']))


#         if self.state['use_gpu']:
#             train_loader.pin_memory = True
#             val_loader.pin_memory = True
#             cudnn.benchmark = True


#             model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()

#             criterion = criterion.cuda()

#         if self.state['evaluate']:
#             with torch.no_grad():
#                 self.validate(val_loader, model, criterion)
#             return

#         # TODO define optimizer

#         for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
#             self.state['epoch'] = epoch
#             lr = self.adjust_learning_rate(optimizer)
#             print('lr:{:.5f}'.format(lr))

#             # train for one epoch
#             self.train(train_loader, model, criterion, optimizer, epoch)
#             # evaluate on validation set
#             with torch.no_grad():
#                 prec1 = self.validate(val_loader, model, criterion)

#             # remember best prec@1 and save checkpoint
#             is_best = prec1 > self.state['best_score']
#             self.state['best_score'] = max(prec1, self.state['best_score'])
#             self.save_checkpoint({
#                 'epoch': epoch + 1,
#                 'arch': self._state('arch'),
#                 'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
#                 'best_score': self.state['best_score'],
#             }, is_best)

#             print(' *** best={best:.3f}'.format(best=self.state['best_score']))
#         self.state['log_path'].close()
#         return self.state['best_score']

#     def train(self, data_loader, model, criterion, optimizer, epoch):

#         # switch to train mode
#         model.train()

#         self.on_start_epoch(True, model, criterion, data_loader, optimizer)

#         if self.state['use_pb']:
#             data_loader = tqdm(data_loader, desc='Training')

#         end = time.time()
#         for i, (input, target) in enumerate(data_loader):
#             # measure data loading time
#             self.state['iteration'] = i
#             self.state['data_time_batch'] = time.time() - end
#             self.state['data_time'].add(self.state['data_time_batch'])

#             self.state['input'] = input
#             self.state['target'] = target

#             self.on_start_batch(True, model, criterion, data_loader, optimizer)

#             if self.state['use_gpu']:
#                 self.state['target'] = self.state['target'].cuda()
#             self.on_forward(True, model, criterion, data_loader, optimizer)

#             # measure elapsed time
#             self.state['batch_time_current'] = time.time() - end
#             self.state['batch_time'].add(self.state['batch_time_current'])
#             end = time.time()
#             # measure accuracy
#             self.on_end_batch(True, model, criterion, data_loader, optimizer)

#         self.on_end_epoch(True, model, criterion, data_loader, optimizer)

#     def validate(self, data_loader, model, criterion):

#         # switch to evaluate mode
#         model.eval()

#         self.on_start_epoch(False, model, criterion, data_loader)

#         if self.state['use_pb']:
#             data_loader = tqdm(data_loader, desc='Test')

#         end = time.time()
#         for i, (input, target) in enumerate(data_loader):
#             # measure data loading time
#             self.state['iteration'] = i
#             self.state['data_time_batch'] = time.time() - end
#             self.state['data_time'].add(self.state['data_time_batch'])

#             self.state['input'] = input
#             self.state['target'] = target

#             self.on_start_batch(False, model, criterion, data_loader)

#             if self.state['use_gpu']:
#                 self.state['target'] = self.state['target'].cuda()

#             self.on_forward(False, model, criterion, data_loader)

#             # measure elapsed time
#             self.state['batch_time_current'] = time.time() - end
#             self.state['batch_time'].add(self.state['batch_time_current'])
#             end = time.time()
#             # measure accuracy
#             self.on_end_batch(False, model, criterion, data_loader)

#         score = self.on_end_epoch(False, model, criterion, data_loader)

#         return score

#     def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
#         if self._state('save_model_path') is not None:
#             filename_ = filename
#             filename = os.path.join(self.state['save_model_path'], filename_)
#             if not os.path.exists(self.state['save_model_path']):
#                 os.makedirs(self.state['save_model_path'])
#         print('save model {filename}'.format(filename=filename))
#         torch.save(state, filename)
#         if is_best:
#             filename_best = 'model_best.pth.tar'
#             if self._state('save_model_path') is not None:
#                 filename_best = os.path.join(self.state['save_model_path'], filename_best)
#             shutil.copyfile(filename, filename_best)
#             if self._state('save_model_path') is not None:
#                 if self._state('filename_previous_best') is not None:
#                     os.remove(self._state('filename_previous_best'))
#                 filename_best = os.path.join(self.state['save_model_path'], 'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
#                 shutil.copyfile(filename, filename_best)
#                 self.state['filename_previous_best'] = filename_best

#     def adjust_learning_rate(self, optimizer):
#         """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#         # lr = args.lr * (0.1 ** (epoch // 30))
#         decay = 0.1 ** (sum(self.state['epoch'] >= np.array(self.state['epoch_step'])))
#         lr = self.state['lr'] * decay
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#         return lr


# class MultiLabelEngine(Engine):
#     def __init__(self, state):
#         Engine.__init__(self, state)
#         if self._state('difficult_examples') is None:
#             self.state['difficult_examples'] = False
#         self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

#     def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
#         self.state['meter_loss'].reset()
#         self.state['batch_time'].reset()
#         self.state['data_time'].reset()
#         self.state['ap_meter'].reset()

#     def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
#         map = 100 * self.state['ap_meter'].value().mean()
#         loss = self.state['meter_loss'].value()[0]
#         OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
#         OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
#         if display:
#             if training:
#                 print('Epoch: [{0}]\t'
#                       'Loss {loss:.4f}\t'
#                       'mAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map))
#                 print('OP: {OP:.4f}\t'
#                       'OR: {OR:.4f}\t'
#                       'OF1: {OF1:.4f}\t'
#                       'CP: {CP:.4f}\t'
#                       'CR: {CR:.4f}\t'
#                       'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
#                 self.state['log_path'].write('Epoch: [{0}]\t'
#                       'Loss {loss:.4f}\t'
#                       'mAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map)+'\n')
#                 self.state['log_path'].write('OP: {OP:.4f}\t'
#                       'OR: {OR:.4f}\t'
#                       'OF1: {OF1:.4f}\t'
#                       'CP: {CP:.4f}\t'
#                       'CR: {CR:.4f}\t'
#                       'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1)+'\n')
#             else:
#                 print('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map))
#                 print('OP: {OP:.4f}\t'
#                       'OR: {OR:.4f}\t'
#                       'OF1: {OF1:.4f}\t'
#                       'CP: {CP:.4f}\t'
#                       'CR: {CR:.4f}\t'
#                       'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
#                 print('OP_3: {OP:.4f}\t'
#                       'OR_3: {OR:.4f}\t'
#                       'OF1_3: {OF1:.4f}\t'
#                       'CP_3: {CP:.4f}\t'
#                       'CR_3: {CR:.4f}\t'
#                       'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))
#                 self.state['log_path'].write('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map)+'\n')
#                 self.state['log_path'].write('OP: {OP:.4f}\t'
#                       'OR: {OR:.4f}\t'
#                       'OF1: {OF1:.4f}\t'
#                       'CP: {CP:.4f}\t'
#                       'CR: {CR:.4f}\t'
#                       'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1)+'\n')
#                 self.state['log_path'].write('OP_3: {OP:.4f}\t'
#                       'OR_3: {OR:.4f}\t'
#                       'OF1_3: {OF1:.4f}\t'
#                       'CP_3: {CP:.4f}\t'
#                       'CR_3: {CR:.4f}\t'
#                       'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k)+'\n')
        

#         return map

#     def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

#         self.state['target_gt'] = self.state['target'].clone()
#         self.state['target'][self.state['target'] == 0] = 1
#         self.state['target'][self.state['target'] == -1] = 0

#         input = self.state['input']
#         self.state['feature'] = input[0]
#         self.state['out'] = input[1]

#     def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

#         self.state['loss_batch'] = self.state['loss'].item()
#         self.state['meter_loss'].add(self.state['loss_batch'])
#         # measure mAP
#         self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])

#         if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
#             loss = self.state['meter_loss'].value()[0]
#             batch_time = self.state['batch_time'].value()[0]
#             data_time = self.state['data_time'].value()[0]
#             if training:
#                 print('Epoch: [{0}][{1}/{2}]\t'
#                       'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
#                       'Data {data_time_current:.3f} ({data_time:.3f})\t'
#                       'Loss {loss_current:.4f} ({loss:.4f})'.format(
#                     self.state['epoch'], self.state['iteration'], len(data_loader),
#                     batch_time_current=self.state['batch_time_current'],
#                     batch_time=batch_time, data_time_current=self.state['data_time_batch'],
#                     data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
#             else:
#                 print('Test: [{0}/{1}]\t'
#                       'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
#                       'Data {data_time_current:.3f} ({data_time:.3f})\t'
#                       'Loss {loss_current:.4f} ({loss:.4f})'.format(
#                     self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
#                     batch_time=batch_time, data_time_current=self.state['data_time_batch'],
#                     data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

#     def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
#         feature_var = self.state['feature']
#         target_var = self.state['target']
#         #print(type(feature_var),feature_var.size())
#         # compute output
#         self.state['output'] = model(feature_var)
#         self.state['loss'] = criterion(self.state['output'], target_var) 

#         if training:
#             optimizer.zero_grad()
#             self.state['loss'].backward()
#             nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
#             optimizer.step()

    



class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # self.save_to_mat()
        # self.save_to_np()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):
        if target.sum() == 0:
            return 0
        else:
            # sort examples
            sorted, indices = torch.sort(output, dim=0, descending=True)

            # Computes prec@i
            pos_count = 0.
            total_count = 0.
            precision_at_i = 0.
            for i in indices:
                label = target[i]
                if difficult_examples and label == 0:
                    continue
                if label == 1:
                    pos_count += 1
                total_count += 1
                if label == 1:
                    precision_at_i += pos_count / total_count
            precision_at_i /= pos_count
            return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)


    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)
            Nc[k] = np.sum(targets * (scores >= 0))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1

