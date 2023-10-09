from http.client import NotConnected
from typing import Dict
from dataclasses import dataclass
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd.functional import *
import jiant.tasks.evaluate as evaluate
import jiant.utils.torch_utils as torch_utils
from jiant.proj.main.modeling.primary import JiantModel, wrap_jiant_forward
from jiant.shared.constants import PHASE
from jiant.shared.runner import get_train_dataloader_from_cache, get_eval_dataloader_from_cache
from jiant.utils.display import maybe_tqdm
from jiant.utils.python.datastructures import InfiniteYield, ExtendedDataClassMixin
from my_container_setup import JiantTaskContainer

def complex_backpropagate(loss, optimizer, model, fp16, n_gpu, gradient_accumulation_steps, max_grad_norm, retain_graph=False):
    if n_gpu > 1:
        loss = loss.mean() 
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps
    if fp16:  # noinspection PyUnresolvedReferences,PyPackageRequirements
        from apex import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
    else:
        loss.backward(retain_graph=retain_graph)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    return loss

@dataclass
class RunnerParameters(ExtendedDataClassMixin):
    local_rank: int
    n_gpu: int
    fp16: bool
    max_grad_norm: float

@dataclass
class TrainState(ExtendedDataClassMixin):
    global_steps: int
    task_steps: Dict[str, int]

    @classmethod
    def from_task_name_list(cls, task_name_list):
        return cls(global_steps=0, task_steps={task_name: 0 for task_name in task_name_list})

    def step(self, task_name):
        self.task_steps[task_name] += 1
        self.global_steps += 1

# Maple for hvp
# Following are utilities to make nn.Module functional
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def get_parms(mod):
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    parms = []
    names = []
    for name, p in list(mod.named_parameters()):
        parms.append(copy.deepcopy(p))
        del_attr(mod, name.split("."))
        names.append(name)
    return parms, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

class JiantRunner:
    def __init__(
        self,
        jiant_task_container: JiantTaskContainer,
        jiant_model: JiantModel,
        optimizer_scheduler,
        device,
        rparams: RunnerParameters,
        log_writer,
    ):
        self.jiant_task_container = jiant_task_container
        self.jiant_model = jiant_model
        self.optimizer_scheduler = optimizer_scheduler
        self.device = device
        self.rparams = rparams
        self.log_writer = log_writer
        self.model = self.jiant_model

    def run_train(self):
        for _ in self.run_train_context():
            pass

    def run_train_context(self, verbose=True):
        train_dataloader_dict = self.get_train_dataloader_dict()
        train_state = TrainState.from_task_name_list(
            self.jiant_task_container.task_run_config.train_task_list
        )
        pbar = maybe_tqdm(
            range(self.jiant_task_container.global_train_config.max_steps),  # 4,810,000
            desc = "Training",
            verbose = verbose,
        )
        for _ in pbar:
            self.run_train_step(
                train_dataloader_dict=train_dataloader_dict, train_state=train_state,
                pbar = pbar
            )
            yield train_state

    def resume_train_context(self, train_state, verbose=True):
        train_dataloader_dict = self.get_train_dataloader_dict()
        start_position = train_state.global_steps
        pbar = maybe_tqdm(
            range(start_position, self.jiant_task_container.global_train_config.max_steps),
            desc="Training",
            initial=start_position,
            total=self.jiant_task_container.global_train_config.max_steps,
            verbose=verbose,
        )
        for _ in pbar:
            self.run_train_step(
                train_dataloader_dict=train_dataloader_dict, train_state=train_state,
                pbar = pbar
            )
            yield train_state

    def run_train_step(self, train_dataloader_dict: dict, train_state: TrainState, pbar):
        self.jiant_model.train()
        task_name, task = self.jiant_task_container.task_sampler.pop()
        task_specific_config = self.jiant_task_container.task_specific_configs[task_name]
        loss_val = 0
        for i in range(task_specific_config.gradient_accumulation_steps):
            batch, batch_metadata = train_dataloader_dict[task_name].pop()
            batch = batch.to(self.device)
            model_output = wrap_jiant_forward(
                jiant_model=self.jiant_model, batch=batch, task=task, compute_loss=True)
            loss = self.complex_backpropagate(
                loss = model_output.loss,
                gradient_accumulation_steps = task_specific_config.gradient_accumulation_steps,
                #retain_graph = retain_graph
            )
            loss_val += loss.item()
        burnin_steps = int(self.user_mode['burnin_steps'])
        thin_steps = int(self.user_mode['thin_steps'])
        noise_discount = float(self.user_mode['nd'])

        if not hasattr(self, "bs_step"):
            self.bs_step = 0
        self.bs_step += 1

        warmup = True if self.bs_step < int(self.user_mode['warmup_steps']) else False
        if warmup == False:
            if self.bs_step == burnin_steps:
                self.logger.info('(leaving burnin period) start collecting lambda samples')
                with torch.no_grad():
                    lamb_vec = nn.utils.parameters_to_vector(
                        self.sampler._transform_positive(self.sampler.lamb_))
                    self.post_lamb_mom1 = lamb_vec*1.0
                    self.post_lamb_mom2 = lamb_vec**2
                self.post_lamb_cnt = 1

            lamb_layers = self.sampler._transform_positive(self.sampler.lamb_)
            N = self.sampler.N
            lr_others = self.sampler.lr
            with torch.no_grad():
                grad_lamb_layers = [
                    -((self.sampler.alp-1)/lamb - self.sampler.bet) / N for lamb in lamb_layers]

                for (pn, p), (zn, z) in zip(self.jiant_model.named_parameters(), self.zero.named_parameters()):
                    assert pn == zn
                    for pgidx in range(len(self.optimizer_scheduler.optimizer.param_groups)):
                        if pn in self.optimizer_scheduler.optimizer.param_groups[pgidx]['names']:
                            break
                    if pgidx == len(self.optimizer_scheduler.optimizer.param_groups):
                        raise ValueError
                    lr = self.optimizer_scheduler.optimizer.param_groups[pgidx]['lr']
                    if lr>0 and p.grad is not None:
                        p.grad += np.sqrt(2/(N*lr)) * torch.randn_like(p) * noise_discount
                    lidx = self.sampler.layer_names.index(pn)
                    lamb = lamb_layers[lidx]
                    grad_lamb_layers[lidx] += -1 * ((p-z).abs()/(lamb**2) - 1/lamb) / N 
                    if p.grad is not None:
                        p.grad += -(-torch.sign(p-z) / lamb) / N 

                for lidx, lamb in enumerate(lamb_layers):
                    self.sampler.lamb_[lidx].grad = grad_lamb_layers[lidx] + np.sqrt(2/(N*lr_others)) * torch.randn_like(lamb) * noise_discount
            
            if hasattr(self, "post_lamb_cnt") and self.post_lamb_cnt > 0 and self.bs_step % thin_steps == 0:
                self.logger.info('(post-burnin) accumulate lambda samples')
                with torch.no_grad():
                    lamb_vec = nn.utils.parameters_to_vector(self.sampler._transform_positive(self.sampler.lamb_))
                    self.post_lamb_mom1 = (lamb_vec + self.post_lamb_cnt*self.post_lamb_mom1) / (self.post_lamb_cnt+1)
                    self.post_lamb_mom2 = (lamb_vec**2 + self.post_lamb_cnt*self.post_lamb_mom2) / (self.post_lamb_cnt+1)
                self.post_lamb_cnt += 1
            self.sampler.optimizer.step()
            self.sampler.optimizer.zero_grad()
        
        self.optimizer_scheduler.step()
        self.optimizer_scheduler.optimizer.zero_grad()
        train_state.step(task_name=task_name)
        entry = {
            "task": task_name,
            "task_step": train_state.task_steps[task_name],
            "global_step": train_state.global_steps,
            "loss_val": loss_val / task_specific_config.gradient_accumulation_steps,
        }
        self.log_writer.write_entry(
            "loss_train",
            entry,
        )
        pbar.set_postfix({'loss': loss_val / task_specific_config.gradient_accumulation_steps})

    def run_val(self, task_name_list, use_subset=None, return_preds=False, verbose=True, phase = "val"):
        self.logger.info('---- Model stats (current): \n%s' % (self.sampler.print_stats(),))
        if hasattr(self, "post_lamb_mom1"):
            internal = self.post_lamb_mom1
            internal_str = '(avg) %.6f, (min) %.6f, (max) %.6f\n' % (
                internal.mean().item(), internal.min().item(), internal.max().item(), 
            )
            qp = np.quantile(internal.to('cpu').detach().numpy(), [0.9, 0.99, 0.999])
            internal_str += '            (median) %.6f, (0.9-tile) %.6f, (0.99-tile) %.6f, (0.999-tile) %.6f\n' % (
                internal.median().item(), qp[0], qp[1], qp[2]
            )
            ret_str = '    lambda: %s\n' % (internal_str,)
            self.logger.info('---- Model stats (running avg): \n%s' % (ret_str,))
        
        print("Log Dir:", self.log_writer.tb_writer.logdir)
        
        evaluate_dict = {}

        val_dataloader_dict = self.get_val_dataloader_dict(
            task_name_list=task_name_list, use_subset=use_subset, phase = phase
        )
        
        val_labels_dict = self.get_val_labels_dict(
            task_name_list=task_name_list, use_subset=use_subset, label_phase = phase
        )
        
        emodel = self.jiant_model
        
        for task_name in task_name_list:
            task = self.jiant_task_container.task_dict[task_name]
            evaluate_dict[task_name] = run_val(
                val_dataloader = val_dataloader_dict[task_name],
                val_labels = val_labels_dict[task_name],
                jiant_model = emodel,
                task = task,
                device = self.device,
                local_rank = self.rparams.local_rank,
                return_preds = return_preds,
                verbose = verbose,
                tag = phase,#maple
                user_mode = self.user_mode,
            )

        return evaluate_dict

    def run_test(self, task_name_list, verbose=True):        
        evaluate_dict = {}
        
        test_dataloader_dict = self.get_test_dataloader_dict()
        
        for task_name in task_name_list:
            task = self.jiant_task_container.task_dict[task_name]
            evaluate_dict[task_name] = run_test(
                test_dataloader = test_dataloader_dict[task_name],
                jiant_model = self.jiant_model,
                task = task,
                device = self.device,
                local_rank = self.rparams.local_rank,
                verbose = verbose,
            )
        
        return evaluate_dict


    def get_train_dataloader_dict(self):
        # Not currently supported distributed parallel
        train_dataloader_dict = {}
        for task_name in self.jiant_task_container.task_run_config.train_task_list:
            task = self.jiant_task_container.task_dict[task_name]
            train_cache = self.jiant_task_container.task_cache_dict[task_name]["train"]
            train_batch_size = self.jiant_task_container.task_specific_configs[
                task_name
            ].train_batch_size
            train_dataloader_dict[task_name] = InfiniteYield(
                get_train_dataloader_from_cache(
                    train_cache=train_cache, task=task, train_batch_size=train_batch_size,
                )
            )
        return train_dataloader_dict

    def _get_eval_dataloader_dict(self, phase, task_name_list, use_subset=False):
        val_dataloader_dict = {}
        for task_name in task_name_list:
            task = self.jiant_task_container.task_dict[task_name]
            eval_cache = self.jiant_task_container.task_cache_dict[task_name][phase]
            task_specific_config = self.jiant_task_container.task_specific_configs[task_name]
            val_dataloader_dict[task_name] = get_eval_dataloader_from_cache(
                eval_cache=eval_cache,
                task=task,
                eval_batch_size=task_specific_config.eval_batch_size,
                subset_num=task_specific_config.eval_subset_num if use_subset else None,
            )
        return val_dataloader_dict

    def get_val_dataloader_dict(self, task_name_list, use_subset=False, phase = "val"):
        return self._get_eval_dataloader_dict(
            phase, task_name_list=task_name_list, use_subset=use_subset,
        )

    def get_val_labels_dict(self, task_name_list, use_subset=False, label_phase = "val"):
        val_labels_dict = {}
        for task_name in task_name_list:
            task_specific_config = self.jiant_task_container.task_specific_configs[task_name]
            val_labels_cache = self.jiant_task_container.task_cache_dict[task_name][label_phase + "_labels"]
            val_labels = val_labels_cache.get_all()
            if use_subset:
                val_labels = val_labels[: task_specific_config.eval_subset_num]
            val_labels_dict[task_name] = val_labels
        return val_labels_dict

    def get_test_dataloader_dict(self):
        return self._get_eval_dataloader_dict(
            task_name_list=self.jiant_task_container.task_run_config.test_task_list,
            phase=PHASE.TEST,
        )

    def complex_backpropagate(self, loss, gradient_accumulation_steps, retain_graph = False):
        return complex_backpropagate(
            loss = loss,
            optimizer = self.optimizer_scheduler.optimizer,
            model = self.jiant_model,
            fp16 = self.rparams.fp16,
            n_gpu = self.rparams.n_gpu,
            gradient_accumulation_steps = gradient_accumulation_steps,
            max_grad_norm = self.rparams.max_grad_norm,
            retain_graph = retain_graph
        )

    def get_runner_state(self):
        # TODO: Add fp16  (issue #1186)
        state = {
            "model": torch_utils.get_model_for_saving(self.jiant_model).state_dict(),
            "optimizer": self.optimizer_scheduler.optimizer.state_dict(),
        }
        return state

    def load_state(self, runner_state):
        torch_utils.get_model_for_saving(self.jiant_model).load_state_dict(runner_state["model"])
        self.optimizer_scheduler.optimizer.load_state_dict(runner_state["optimizer"])


class CheckpointSaver:
    def __init__(self, metadata, save_path):
        self.metadata = metadata
        self.save_path = save_path
    def save(self, runner_state: dict, metarunner_state: dict):
        to_save = {
            "runner_state": runner_state,
            "metarunner_state": metarunner_state,
            "metadata": self.metadata,
        }
        torch_utils.safe_save(to_save, self.save_path)

def run_val(
    val_dataloader,
    val_labels,
    jiant_model: JiantModel,
    task,
    device,
    local_rank,
    return_preds=False,
    verbose=True,
    tag="Val",
    user_mode = None,
):
    if not local_rank == -1:
        return

    jiant_model.eval()

    total_eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task=task)
    eval_accumulator = evaluation_scheme.get_accumulator()

    for step, (batch, batch_metadata) in enumerate(
        maybe_tqdm(val_dataloader, desc=f"Eval ({task.name}, {tag})", verbose=verbose)
    ):
        batch = batch.to(device)
        with torch.no_grad():
            model_output = wrap_jiant_forward(
                jiant_model=jiant_model, batch=batch, task=task, compute_loss=True)
        batch_logits = model_output.logits.detach().cpu().numpy()
        batch_loss = model_output.loss.mean().item()
        total_eval_loss += batch_loss
        eval_accumulator.update(
            batch_logits = batch_logits,
            batch_loss = batch_loss,
            batch = batch,
            batch_metadata = batch_metadata,
        )
        nb_eval_examples += len(batch)
        nb_eval_steps += 1

    eval_loss = total_eval_loss / nb_eval_steps

    tokenizer = (
        jiant_model.tokenizer
        if not torch_utils.is_data_parallel(jiant_model)
        else jiant_model.module.tokenizer
    )

    output = {
        "accumulator": eval_accumulator,
        "loss": eval_loss,
        "metrics": evaluation_scheme.compute_metrics_from_accumulator(
            task=task, accumulator=eval_accumulator, labels=val_labels, tokenizer=tokenizer,
        ),
    }

    if return_preds:
        output["preds"] = evaluation_scheme.get_preds_from_accumulator(
            task=task, accumulator=eval_accumulator,
        )

    return output

def run_test(
    test_dataloader,
    jiant_model: JiantModel,
    task,
    device,
    local_rank,
    verbose=True,
    return_preds=True,
):

    if not local_rank == -1:
        return

    jiant_model.eval()

    evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task=task)
    eval_accumulator = evaluation_scheme.get_accumulator()

    for step, (batch, batch_metadata) in enumerate(
        maybe_tqdm(test_dataloader, desc=f"Eval ({task.name}, Test)", verbose=verbose)
    ):
        batch = batch.to(device)
        with torch.no_grad():
            model_output = wrap_jiant_forward(
                jiant_model=jiant_model, batch=batch, task=task, compute_loss=False)
        batch_logits = model_output.logits.detach().cpu().numpy()
        eval_accumulator.update(
            batch_logits=batch_logits, batch_loss=0, batch=batch, batch_metadata=batch_metadata,
        )

    output = {
        "accumulator": eval_accumulator,
    }

    if return_preds:
        output["preds"] = evaluation_scheme.get_preds_from_accumulator(
            task=task, accumulator=eval_accumulator,
        )

    return output
