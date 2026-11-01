from pathlib import Path
import hashlib
import importlib.util
import time

HERE=Path(__file__).resolve().parent
ROOT = HERE
BACKEND = HERE / 'yelp_train_strict_bmm.py'
METADATA = HERE.parent / 'input/saint_canonical_coo_metadata_v1.py'
EXPECTED_METADATA_SHA256='aa3ebca08cd03d69c4c59338da20ba3b1bb88dfc1e31ab7858e86c2ef78b69c2'
VERSION='Yelp-canonical-positive-CPU-metadata-original-public-bmm-v1'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

metadata=load(METADATA,'Yelp_original_canonical_positive_converter')
original=load(BACKEND,'Yelp_original_strict_public_backend_unchanged')
ROUTE=original.ROUTE
configure_before_torch=original.configure_before_torch
_qualification=None

def specification():
    return dict(original.specification(),canonical_CPU_metadata=dict(version=VERSION,
        helper=str(METADATA),helper_sha256=EXPECTED_METADATA_SHA256,
        actual_public_bmm_unchanged=True,custom_singleton_DSO_loaded=False,
        only_fresh_original_producer_sorted_unique_positive_normal_FP32=True,
        all_other_inputs_use_original_converter=True,metadata_installed_after_ContextIndex=True,
        every_value_scan_install_and_restore_inside_complete_outer_TRAIN=True,
        no_adjacency_cache_or_dtype_or_reduction_change=True,
        original_transpose_coalesce_and_public_batch_D2H_retained=True))

def adapt(parent):
    base=original.adapt(parent)
    class CanonicalPublicTraining(base):
        def __init__(self,model,method):
            super().__init__(model,method)
            if method!='lateA':raise ValueError('Only original S is changed; compatible D/G retain their original sources')
            self._canonical_counts=dict(calls=0,canonical_positive=0,fallback=0,epochs=0,failed_epochs=0)
            self._canonical_live=False
            self._canonical_extra_outer_seconds=0.

        def work(self):
            result=super().work()
            result['canonical_public_CPU_metadata']=dict(specification()['canonical_CPU_metadata'],
                **self._canonical_counts,installed=self._canonical_live,
                extra_install_restore_outer_seconds=self._canonical_extra_outer_seconds,
                closed=not self._canonical_live and self._strict_close_calls==1)
            return result

        def verify_final(self,steps):
            result=super().verify_final(steps)
            if (self._canonical_live or self._canonical_counts['calls']!=8100
                    or self._canonical_counts['epochs']!=75 or self._canonical_counts['failed_epochs']
                    or self._canonical_counts['canonical_positive']+self._canonical_counts['fallback']!=8100):
                raise RuntimeError('Every original TRAIN converter must be accounted before close')
            return result
    return CanonicalPublicTraining

def train_epoch(parent,minibatch,engine,warm_initial,counters,audit,synchronize,before_epoch=None,clock=time.perf_counter):
    
    
    started=clock()
    if engine._canonical_live:raise RuntimeError('No nested canonical converter scope')
    engine._canonical_live=True
    installed=False;success=False;inner=None
    try:
        metadata.install(minibatch);installed=True
        inner=original.train_epoch(parent,minibatch,engine,warm_initial,counters,audit,synchronize,before_epoch=before_epoch,clock=clock)
        success=True
    finally:
        try:
            if installed:
                work=metadata.close(minibatch)
                for key in('calls','canonical_positive','fallback'):engine._canonical_counts[key]+=work[key]
                engine._canonical_counts['epochs']+=1
                if not success:engine._canonical_counts['failed_epochs']+=1
        finally:engine._canonical_live=False
    elapsed=clock()-started
    extra=elapsed-inner
    if extra<0:raise RuntimeError('Canonical enclosing epoch duration cannot omit original time')
    engine._canonical_extra_outer_seconds+=extra
    engine._strict_outer_seconds+=extra
    return elapsed
