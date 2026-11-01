import importlib.util
import inspect
from pathlib import Path
import sys
import textwrap
from types import FunctionType, MethodType, SimpleNamespace

HERE=Path(__file__).resolve().parent
PROTOTYPE = HERE / 'native_adam_graph_prototype.py'
VERSION='T456-Yelp-original-native-fusedAdam-graph-realAPI-v1'
REUSED_FILES=[PROTOTYPE]

def _prototype_module():
    key='T456_frozen_native_Adam_graph_prototype'
    if key not in sys.modules:
        spec=importlib.util.spec_from_file_location(key,PROTOTYPE)
        module=importlib.util.module_from_spec(spec)
        sys.modules[key]=module
        spec.loader.exec_module(module)
    return sys.modules[key]

_prototype=_prototype_module()

def _derived_method(function,old,new):
    source=textwrap.dedent(inspect.getsource(function))
    if source.count(old)!=1:
        raise RuntimeError('Frozen native Adam prototype anchor changed: '+old)
    namespace=dict(function.__globals__)
    exec(compile(source.replace(old,new),__file__+'::'+function.__name__,'exec'),namespace)
    return namespace[function.__name__]

_constructor=_derived_method(_prototype.NativeAdamGraphs.__init__,
    "    if optimizer._optimizer_step_pre_hooks or optimizer._optimizer_step_post_hooks:\n"
    "        raise ValueError('This finite prototype does not change or replay live hooks')\n",
    '')
_cold_capture=_derived_method(_prototype.NativeAdamGraphs._new,
    'try:shadow.step()', 'try:self._capture_native_step(shadow)')

class NativeAdamGraphs(_prototype.NativeAdamGraphs):

    def __init__(self,optimizer,*,head_parameters,body_parameters,inactive_parameters=()):
        import torch
        
        
        if (getattr(optimizer.step,'__self__',None) is not optimizer
                or getattr(optimizer.step,'__func__',None) is not torch.optim.Adam.step):
            raise ValueError('Original native Adam.step method required before install')
        self._had_instance_step='step' in optimizer.__dict__
        _constructor(self,optimizer)
        head=tuple(head_parameters);body=tuple(body_parameters);inactive=tuple(inactive_parameters)
        head_ids={id(p) for p in head};body_ids={id(p) for p in body};inactive_ids={id(p) for p in inactive}
        groups=tuple(tuple(g['params']) for g in optimizer.param_groups)
        members=tuple(p for group in groups for p in group)
        if (len(head)!=2 or len(head_ids)!=2 or len(body_ids)!=len(body)
                or len(inactive_ids)!=len(inactive) or head_ids & body_ids
                or (head_ids|body_ids) & inactive_ids
                or head_ids|body_ids|inactive_ids!={id(p) for p in members}
                or len(members)!=len(head)+len(body)+len(inactive)
                or any(p.grad is not None for p in inactive)):
            raise ValueError('Exact original disjoint Head/body optimizer membership required')
        self._expected_keys={
            tuple(tuple(id(p) for p in group if id(p) in active) for group in groups)
            for active in (head_ids,head_ids|body_ids,body_ids)}
        if len(self._expected_keys)!=3:
            raise ValueError('Original C, C+body and body supports required')
        self._state_owner=optimizer.state
        self._inactive_parameters=inactive
        self._groups_owner=optimizer.param_groups
        self._group_owners=tuple(optimizer.param_groups)
        self._parameter_members=tuple(tuple(id(p) for p in group) for group in groups)
        self.real_wrapper_calls=0
        self.closed=False
        self._installed_step=None

    @staticmethod
    def _capture_native_step(shadow):
        import torch
        outer=torch.optim.Adam.step
        
        
        
        if not getattr(outer,'hooked',False) or not hasattr(outer,'__wrapped__'):
            raise RuntimeError('Original official Adam profile-hook wrapper required')
        return outer.__wrapped__(shadow)

    def _new(self,support):
        key=tuple(tuple(id(p) for p in group) for group in support)
        if key not in self._expected_keys:
            raise RuntimeError('An unqualified Adam gradient support was requested')
        return _cold_capture(self,support)

    def install(self):
        from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable
        if self._installed_step is not None or self.closed:
            raise RuntimeError('Install this native API dispatcher exactly once')
        engine=self
        def dispatch(optimizer,closure=None):
            if optimizer is not engine.optimizer or engine.closed:
                raise RuntimeError('Original live Adam owner required')
            engine.real_wrapper_calls+=1
            return engine.step(closure)
        
        
        wrapped=Optimizer.profile_hook_step(_use_grad_for_differentiable(dispatch))
        self._installed_step=MethodType(wrapped,self.optimizer)
        self.optimizer.step=self._installed_step
        return self

    def verify(self,expected_calls):
        optimizer=self.optimizer
        if (self.calls!=expected_calls or self.real_wrapper_calls!=expected_calls
                or self.captures!=3 or set(self.graphs)!=self._expected_keys
                or optimizer.state is not self._state_owner
                or optimizer.param_groups is not self._groups_owner
                or len(optimizer.param_groups)!=len(self._group_owners)
                or any(a is not b for a,b in zip(optimizer.param_groups,self._group_owners))
                or tuple(tuple(id(p) for p in g['params']) for g in optimizer.param_groups)!=self._parameter_members
                or optimizer.step is not self._installed_step or self.closed
                or any(p.grad is not None or optimizer.state.get(p) for p in self._inactive_parameters)):
            raise RuntimeError('Original native Adam API/state/support/ownership differs')
        return dict(self.work(),passed=True)

    def work(self):
        result=super().work()
        result.update(version=VERSION,real_driver_installed=self._installed_step is not None,
            live_original_optimizer_API_wrappers=self.real_wrapper_calls,
            original_local_and_global_pre_post_hooks_preserved=True,
            original_optimizer_step_code_observer_preserved=True,
            private_shadow_capture_bypasses_only_official_profile_hooks=True,
            unused_parameters_with_original_empty_state=len(self._inactive_parameters),
            support_programs='C, C+body, body',closed=self.closed,
            cold_state_capture_and_all_staging_paid=True)
        return result

    def close(self):
        if self.closed:raise RuntimeError('Exactly one native Adam dispatcher close required')
        if self.optimizer.step is not self._installed_step:
            raise RuntimeError('Live native Adam dispatcher was replaced externally')
        if self._had_instance_step:self.optimizer.step=self.original_step
        else:del self.optimizer.step
        self.closed=True

def install_native_adam(optimizer,*,head_parameters,body_parameters,inactive_parameters=()):
    return NativeAdamGraphs(optimizer,head_parameters=head_parameters,
        body_parameters=body_parameters,inactive_parameters=inactive_parameters).install()

def adapted_class(original):
    class NativeAdamGraphTraining(original):
        def __init__(self,model,method):
            super().__init__(model,method)
            self._native_adam_graph=install_native_adam(model.optimizer,
                head_parameters=tuple(self.groups['C'].values()),
                body_parameters=tuple(self.body.values()),
                inactive_parameters=tuple(self.unused.values()))

        def work(self):
            result=super().work()
            result['native_Adam_graph']=self._native_adam_graph.work()
            return result

        def verify_final(self,steps):
            result=super().verify_final(steps)
            self._native_adam_graph.verify(3*self.blocks)
            return result

        def close(self):
            try:return super().close()
            finally:self._native_adam_graph.close()
    return NativeAdamGraphTraining

def adapter_view(original):
    old_adapt=original.adapt
    def adapt(parent,rule):
        inherited=old_adapt(parent,rule)
        return adapted_class(inherited) if rule=='ordered' else inherited
    old_specification=original.specification
    def specification(rule):
        result=dict(old_specification(rule))
        result['native_fused_Adam_graph_real_API']=dict(version=VERSION,enabled=rule=='ordered',
            same_native_optimizer_parameters_state_options_and_clocks=True,
            original_three_Adam_API_supports_and_live_hooks=True,
            native_Float32_kernel_LR_cast_from_original_Python_float=True,
            native_initial_zero_state_only_no_dummy_update=True,
            import_setup_capture_staging_replay_all_paid=True,
            requires_current_full_chain_strict_qualification=True)
        return result
    view=SimpleNamespace(**vars(original))
    view.adapt,view.specification=adapt,specification
    return view
