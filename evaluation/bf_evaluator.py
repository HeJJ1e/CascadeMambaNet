from os import makedirs
from os.path import exists, join
from utils.ply import write_ply
from sklearn.metrics import confusion_matrix
from utils.pointcloud import DataProcessing as DP
from configs.settings import BFConfig as cfg
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

def log_out(out_str, log_f_out):
    log_f_out.write(out_str + '\n')
    log_f_out.flush()
    print(out_str)

class InfiniteDataset(IterableDataset):

    def __init__(self, gen_func):
        self.gen_func = gen_func

    def __iter__(self):
        return self.gen_func()

class ModelTester:

    def __init__(self, model, dataset, restore_snap=None):
        self.Log_file = open('log_test_' + str(dataset.val_split) + '.txt', 'a')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        if restore_snap is not None and exists(restore_snap):
            checkpoint = torch.load(restore_snap, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                checkpoint = checkpoint.get('ema_state_dict') or checkpoint['model_state_dict']
            self.model.load_state_dict(checkpoint, strict=True)
            print('Model restored from ' + restore_snap)
        else:
            raise FileNotFoundError(f'model snapshot does not exist: {restore_snap}')
        self.model.eval()
        self.test_probs = [np.zeros(shape=[l.shape[0], cfg.num_classes], dtype=np.float32) for l in dataset.input_labels['validation']]

    def test(self, model, dataset):
        from engine.bf_pipeline import cascade_collate_fn
        test_smooth = 0.95
        val_proportions = np.zeros(cfg.num_classes, dtype=np.float32)
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.val_labels])
                i += 1
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'val_preds')) if not exists(join(test_path, 'val_preds')) else None
        step_id = 0
        epoch_id = 0
        last_min = -0.5
        val_gen = dataset.get_batch_gen('validation')
        loader = DataLoader(InfiniteDataset(val_gen), batch_size=cfg.val_batch_size, collate_fn=cascade_collate_fn)
        loader_iter = iter(loader)
        log_out('TTA disabled (single-view inference)', self.Log_file)
        with torch.no_grad():
            while True:
                batch_data = next(loader_iter)
                probs = F.softmax(model(batch_data), dim=-1)
                stacked_probs = probs.cpu().numpy()
                stacked_labels = batch_data['labels'].cpu().numpy()
                point_idx = batch_data['input_inds'].cpu().numpy()
                cloud_idx = batch_data['cloud_inds'].cpu().numpy()
                correct = np.sum(np.argmax(stacked_probs, axis=-1) == stacked_labels)
                acc = correct / float(np.prod(np.shape(stacked_labels)))
                if step_id % 50 == 0:
                    print('step ' + str(step_id) + ' acc: ' + str(acc))
                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    p_idx = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][p_idx] = test_smooth * self.test_probs[c_i][p_idx] + (1 - test_smooth) * probs
                step_id += 1
                new_min = np.min(dataset.min_possibility['validation'])
                if last_min + 1 < new_min:
                    log_out('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.Log_file)
                    last_min += 1
                    epoch_id += 1
                    log_out('\nConfusion on sub clouds', self.Log_file)
                    confusion_list = []
                    num_val = len(dataset.input_labels['validation'])
                    for i_test in range(num_val):
                        probs = self.test_probs[i_test]
                        preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                        labels = dataset.input_labels['validation'][i_test]
                        confusion_list += [confusion_matrix(labels, preds, labels=dataset.label_values)]
                    C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-06), 1)
                    IoUs = DP.IoU_from_confusions(C)
                    m_IoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * m_IoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    log_out(s + '\n', self.Log_file)
                    if int(np.ceil(new_min)) % 1 == 0:
                        log_out('\nReproject vote {:d}'.format(int(np.floor(new_min))), self.Log_file)
                        proj_probs_list = []
                        for i_val in range(num_val):
                            proj_idx = dataset.val_proj[i_val]
                            probs = self.test_probs[i_val][proj_idx, :]
                            proj_probs_list += [probs]
                        log_out('Confusion on full clouds', self.Log_file)
                        confusion_list = []
                        for i_test in range(num_val):
                            preds = dataset.label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)
                            labels = dataset.val_labels[i_test]
                            acc = np.sum(preds == labels) / len(labels)
                            log_out(dataset.input_names['validation'][i_test] + ' Acc:' + str(acc), self.Log_file)
                            confusion_list += [confusion_matrix(labels, preds, labels=dataset.label_values)]
                            name = dataset.input_names['validation'][i_test] + '.ply'
                            write_ply(join(test_path, 'val_preds', name), [preds, labels], ['pred', 'label'])
                        C = np.sum(np.stack(confusion_list), axis=0)
                        IoUs = DP.IoU_from_confusions(C)
                        m_IoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * m_IoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        log_out('-' * len(s), self.Log_file)
                        log_out(s, self.Log_file)
                        log_out('-' * len(s) + '\n', self.Log_file)
                        print('finished \n')
                        return
