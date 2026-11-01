from pathlib import Path
from types import SimpleNamespace

VERSION='T456-Yelp-direct-current-Head-mu-clip-Graph-v1'

class HeadQuery:
    def __init__(self, model, parameters, geometry, *, bucket_rows=128, warmups=3):
        self.model=model
        self.parameters=tuple(parameters)
        expected={id(model.classifier.f_lin[0].weight),id(model.classifier.f_bias[0])}
        if len(self.parameters)!=2 or {id(p) for p in self.parameters}!=expected:
            raise ValueError('Original two Head parameter objects in original clip order required')
        self.geometry=geometry
        self.bucket_rows=bucket_rows
        self.warmups=warmups
        self.graphs={}
        self.calls=self.stages=self.staged_rows=self.cold_builds=0
        self.closed=self.failed=False

    def _new(self, rows, late):
        import torch
        dtype,device=self.parameters[0].dtype,self.parameters[0].device
        labels_count=self.model.classifier.f_bias[0].numel()
        owned=dict(
            boundary=torch.zeros((rows,512),dtype=dtype,device=device,requires_grad=True),
            z=torch.zeros((rows,512),dtype=dtype,device=device),
            branch=torch.zeros((rows,1),dtype=torch.bool,device=device),
            labels=torch.zeros((rows,labels_count),dtype=dtype,device=device),
            weights=torch.zeros((rows,),dtype=dtype,device=device),
            grads=tuple(torch.zeros_like(p) for p in self.parameters),
            late=late)
        owned['boundary_gradient']=torch.zeros_like(owned['boundary'])
        owned['handles']=tuple(p.detach() for p in self.parameters)
        for p,g in zip(owned['handles'],owned['grads']):p.grad=g
        original_gradients=tuple(p.grad for p in self.parameters)
        current=torch.cuda.current_stream(device)
        stream=torch.cuda.Stream(device=device)
        stream.wait_stream(current)

        def compute():
            import torch.nn.functional as functional
            boundary=owned['boundary']
            boundary.grad=owned['boundary_gradient']
            boundary.grad.zero_()
            for p,g in zip(self.parameters,owned['grads']):
                p.grad=g
                g.zero_()
            logits=self.model.classifier.f_lin[0](boundary)+self.model.classifier.f_bias[0]
            loss=functional.binary_cross_entropy_with_logits(logits,owned['labels'],
                weight=owned['weights'].unsqueeze(1),
                pos_weight=self.model._r69_late_positive_bce_scalar if late else None,
                reduction='sum')
            loss.backward()
            raw_lambda=boundary.grad
            mu=self.geometry.channel_measurement(owned['z'],boundary,owned['branch'],raw_lambda)
            norm=torch.nn.utils.clip_grad_norm_(owned['handles'],5.)
            return loss,raw_lambda,mu,norm

        try:
            with torch.cuda.stream(stream):
                for _ in range(self.warmups):
                    dummy=compute()
                del dummy
            stream.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                graph.capture_begin(capture_error_mode='global')
                try:
                    output=compute()
                finally:
                    graph.capture_end()
            current.wait_stream(stream)
            owned.update(graph=graph,outputs=output,stream=stream,
                         current_id=current.cuda_stream)
        finally:
            for p,g in zip(self.parameters,original_gradients):p.grad=g
        self.cold_builds+=1
        return owned

    def begin(self, boundary, z, branch, labels, weights, *, late):
        import torch
        if self.closed or self.failed:
            raise RuntimeError('A closed or failed current-Head graph cannot continue')
        rows=boundary.shape[0]
        bucket=((rows+self.bucket_rows-1)//self.bucket_rows)*self.bucket_rows
        key=(bucket,bool(late))
        try:
            if key not in self.graphs:self.graphs[key]=self._new(bucket,bool(late))
            owned=self.graphs[key]
            
            
            with torch.no_grad():
                for target,source in [(owned['boundary'],boundary),(owned['z'],z),
                                      (owned['branch'],branch),(owned['labels'],labels),
                                      (owned['weights'],weights)]:
                    target[:rows].copy_(source)
                    target[rows:].zero_()
            self.current=owned
            self.rows=rows
            self.current_queries=0
            self.stages+=5
            self.staged_rows+=rows
        except BaseException:
            self.failed=True
            raise

    def query(self):
        import torch
        if self.closed or self.failed or self.current_queries>=2:
            raise RuntimeError('Exactly two healthy current Head queries required')
        owned=self.current
        try:
            
            
            owned['graph'].replay()
            for p,g in zip(self.parameters,owned['grads']):p.grad=g
            loss,raw_lambda,mu,_norm=owned['outputs']
            self.calls+=1
            self.current_queries+=1
            
            
            return loss.detach().clone(),raw_lambda[:self.rows],mu.clone()
        except BaseException:
            self.failed=True
            raise

    def work(self):
        return dict(version=VERSION,real_Head_queries=self.calls,cold_graphs=self.cold_builds,
            normal_staging_copies=self.stages,real_rows_staged=self.staged_rows,
            dummy_warmup_queries=self.cold_builds*self.warmups,
            capture_recorded_queries=self.cold_builds,
            native_Adam_inside_graph=False,random_draws_inside_graph=False,
            native_Head_clipping_and_FP64_measurement_inside_graph=True,
            cold_setup_and_all_copies_paid_in_TRAIN=True,
            bitwise_equivalence_claim=False,failed=self.failed,closed=self.closed)

    def close(self):
        self.closed=True

def adapted_class(original):
    class DirectHeadQueryTraining(original):
        def __init__(self,model,method):
            super().__init__(model,method)
            
            
            original_measure=super()._measure
            geometry=original_measure.__func__.__globals__['geometry']
            self._direct_head=HeadQuery(model,tuple(self.groups['C'].values()),geometry)

        def block(self,nodes,adj,weights):
            if self.closed:raise RuntimeError('Training engine already closed')
            model=self.model
            model.train()
            model.optimizer.zero_grad(set_to_none=True)
            self.randomness.begin()
            self.active_cache=True
            losses=[]
            try:
                prefix=self._structured_prefix(nodes,adj)
                labels=model.label_full[nodes]
                self._direct_head.begin(prefix.detach(),self._z,self._unclamped,labels,weights,
                                        late=bool(self.positive_phase.active))
                cotangent=None
                measurements=[]
                for query in range(2):
                    loss,raw_lambda,mu=self._direct_head.query()
                    losses.append(loss)
                    measurements.append(mu)
                    self.structural_queries+=1
                    self.clip_calls['C']+=1
                    if cotangent is None:cotangent=raw_lambda.detach().clone()
                    else:cotangent.add_(raw_lambda)
                    if query==0:self._step()
                cotangent.mul_(.5)
                prefix.backward(cotangent)
                self._clip(self.body_labels)
                self._prepare_pair(measurements)
                gradients={k:p.grad for k,p in self.body.items()}
                self._install_pair(0)
                self._step()
                for k,p in self.body.items():p.grad=gradients[k]
                self._install_pair(1)
                self._step()
                self.prefix_forwards+=1
                self.body_vjps+=1
                self.local_forwards+=1
                self.local_vjps+=1
                self.head_queries+=2
                self.blocks+=1
                self.visits+=2
                self.loss_node_presentations+=2*len(nodes)
                self.prefix_node_presentations+=len(nodes)
                return losses
            finally:
                self._release_structural()
                self.randomness.end()
                self.active_cache=False

        def work(self):
            result=super().work()
            result['direct_Head_query_graph']=self._direct_head.work()
            return result

        def verify_final(self,steps):
            result=super().verify_final(steps)
            if self._direct_head.calls!=2*self.blocks or self._direct_head.failed:
                raise RuntimeError('Original current Head query graph counts changed')
            return result

        def close(self):
            try:return super().close()
            finally:self._direct_head.close()
    return DirectHeadQueryTraining

def adapter_view(original):
    old_adapt=original.adapt
    def adapt(parent,rule):
        inherited=old_adapt(parent,rule)
        return adapted_class(inherited) if rule=='ordered' else inherited
    old_specification=original.specification
    def specification(rule):
        result=dict(old_specification(rule))
        result['direct_Head_query_graph']=dict(version=VERSION,enabled=rule=='ordered',
            same_two_current_Head_queries_and_original_Adam=True,
            same_FP64_measurement_expression_and_native_Head_clip=True,
            padded_FP32_shapes_changed=True,requires_finite_GPU_qualification=True)
        return result
    view=SimpleNamespace(**vars(original))
    view.adapt,view.specification=adapt,specification
    return view
