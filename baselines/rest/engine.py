import time
import torch
from torch_geometric_autoscale import SubgraphLoader, EvalSubgraphLoader, dropout
from torch_geometric_autoscale.models import GCN

def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

class CountedOptimizer:

    def __init__(self, optimizer, counters):
        (self.optimizer, self.counters) = (optimizer, counters)

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.counters['optimizer_updates'] += 1
        return self.optimizer.step(*args, **kwargs)

class NativeIdsLoader:

    def __init__(self, loader, counters):
        (self.loader, self.counters) = (loader, counters)
        self.data = loader.data

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for item in self.loader:
            (data, core, ids, *_) = item
            self.counters['batches'] += 1
            self.counters['core_nodes'] += core
            self.counters['nodes'] += ids.numel()
            self.counters['edges'] += data.adj_t.nnz()
            yield item

class Engine:

    def __init__(self, data, ptr, initial, recipe, device):
        self.device = torch.device(device)
        self.generator = torch.Generator().manual_seed(recipe['seed'])
        self.refresh_generator = torch.Generator().manual_seed(recipe['seed'] + 1)
        self.loader = SubgraphLoader(data, ptr, batch_size=recipe['batch_parts'], shuffle=True, num_workers=0, log=False, generator=self.generator)
        self.refresh_loader = SubgraphLoader(data, ptr, batch_size=recipe['batch_parts'], shuffle=True, num_workers=0, log=False, generator=self.refresh_generator)
        self.eval_loader = EvalSubgraphLoader(data, ptr, batch_size=recipe['batch_parts'], log=False)
        buffer = max((item.n_id.numel() for item in self.eval_loader))
        self.model = GCN(data.num_nodes, data.x.shape[1], recipe['hidden'], int(data.y.max()) + 1, recipe['layers'], dropout=recipe['dropout'], drop_input=False, batch_norm=recipe['batch_norm'], residual=False, linear=False, pool_size=recipe['pool_size'], buffer_size=buffer).to(self.device)
        self.model.load_state_dict(initial, strict=True)
        self.optimizer = torch.optim.Adam([dict(params=self.model.reg_modules.parameters(), weight_decay=0.0), dict(params=self.model.nonreg_modules.parameters(), weight_decay=0.0)], lr=recipe['lr'])
        self.counters = dict(batches=0, core_nodes=0, nodes=0, edges=0, optimizer_updates=0, full_refreshes=0)
        self.rest_counters = dict(batches=0, core_nodes=0, nodes=0, edges=0, forward_only_calls=0, training_label_presentations=0, skipped_unlabeled_training_batches=0)
        self.device_loader = NativeIdsLoader(self.loader, self.counters)
        self.refresh_device_loader = NativeIdsLoader(self.refresh_loader, self.rest_counters)
        self.counted_optimizer = CountedOptimizer(self.optimizer, self.counters)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_function = mini_train
        self._closed = self._terminal_partial_pass = False
        self.epoch = self.intervals = 0
        self.last_logits = self.full_refresh()

    @torch.no_grad()
    def full_refresh(self):
        self.model.eval()
        result = self.model(loader=self.eval_loader)
        self.counters['full_refreshes'] += 1
        return result

    def close(self):
        if not self._closed:
            self.model.pool.synchronize_push()
            self._closed = True

    @torch.no_grad()
    def offline_refresh(self, batch):
        (data, batch_size, *ids) = batch
        self.model.train()
        self.model(data.x.to(self.device), data.adj_t.to(self.device), batch_size, *ids)
        self.rest_counters['forward_only_calls'] += 1

    def epoch_step(self, *, max_updates=None):
        if self._closed:
            raise RuntimeError('REST engine has been closed')
        if self._terminal_partial_pass:
            raise RuntimeError('The exact-budget partial traversal was terminal')
        if max_updates is not None and max_updates <= 0:
            raise ValueError('Use a positive remaining update budget')
        _sync(self.device)
        start = time.perf_counter()
        before = dict(self.counters)
        before_rest = dict(self.rest_counters)
        self.model.train()
        refresh_iterator = iter(self.refresh_device_loader)
        training_iterator = iter(self.device_loader)
        weighted_loss = label_count = 0
        complete_pass = False
        try:
            for (batch_index, batch) in enumerate(training_iterator):
                (data, batch_size, *_) = batch
                labels_in_batch = int(data.train_mask[:batch_size].sum())
                if not labels_in_batch:
                    self.rest_counters['skipped_unlabeled_training_batches'] += 1
                    continue
                self.offline_refresh(next(refresh_iterator))
                loss = self.train_function(self.model, [batch], self.criterion, self.counted_optimizer, 1, grad_norm=None, edge_dropout=0.0)
                weighted_loss += loss * labels_in_batch
                label_count += labels_in_batch
                self.rest_counters['training_label_presentations'] += labels_in_batch
                if max_updates is not None and self.counters['optimizer_updates'] - before['optimizer_updates'] >= max_updates:
                    complete_pass = batch_index + 1 == len(self.loader)
                    if complete_pass and next(training_iterator, None) is not None:
                        raise RuntimeError('Training loader length contract differs')
                    break
            else:
                complete_pass = True
            if self.rest_counters['forward_only_calls'] - before_rest['forward_only_calls'] == len(self.refresh_loader):
                if next(refresh_iterator, None) is not None:
                    raise RuntimeError('Refresh loader length contract differs')
            _sync(self.device)
            after_train = time.perf_counter()
            self.last_logits = self.full_refresh()
            _sync(self.device)
            end = time.perf_counter()
        except BaseException:
            self.close()
            raise
        finally:
            for iterator in (training_iterator, refresh_iterator):
                close = getattr(iterator, 'close', None)
                if close is not None:
                    close()
        self.epoch += int(complete_pass)
        self.intervals += 1
        self._terminal_partial_pass = not complete_pass
        return {'complete_epochs': self.epoch, 'refresh_intervals': self.intervals, 'complete_training_pass': complete_pass, 'terminal_partial_pass': not complete_pass, 'train_seconds_including_REST': after_train - start, 'required_full_refresh_seconds': end - after_train, 'training_label_presentations': label_count, 'author_training_loss': weighted_loss / label_count if label_count else None, 'training_counter_delta': {k: self.counters[k] - before[k] for k in before}, 'REST_counter_delta': {k: self.rest_counters[k] - before_rest[k] for k in before_rest}, 'counters': dict(self.counters), 'REST_counters': dict(self.rest_counters)}

def mini_train(model, loader, criterion, optimizer, max_steps, grad_norm=None, edge_dropout=0.0):
    model.train()
    total_loss = total_examples = 0
    for (i, (batch, batch_size, *args)) in enumerate(loader):
        x = batch.x.to(model.device)
        adj_t = batch.adj_t.to(model.device)
        y = batch.y[:batch_size].to(model.device)
        train_mask = batch.train_mask[:batch_size].to(model.device)
        if train_mask.sum() == 0:
            continue
        adj_t = dropout(adj_t, p=edge_dropout)
        optimizer.zero_grad()
        out = model(x, adj_t, batch_size, *args)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
        total_loss += float(loss) * int(train_mask.sum())
        total_examples += int(train_mask.sum())
        if i + 1 >= max_steps and i + 1 < len(loader):
            break
    return total_loss / total_examples
