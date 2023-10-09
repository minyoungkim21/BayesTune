import os
import datetime
import copy
import numpy as np
import torch
import jiant.proj.main.modeling.model_setup as jiant_model_setup
import jiant.shared.initialization as initialization
import jiant.shared.distributed as distributed
import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf
import my_jiant_runner2
import my_jiant_metarunner
import my_writer
import my_model_setup
import my_zlog
import my_container_setup
import my_sampler
import my_utils

@zconf.run_config
class RunConfiguration(zconf.RunConfig):

    # === Required parameters === #
    jiant_task_container_config_path = zconf.attr(type=str, required=True)
    output_dir = zconf.attr(type=str, required=True)

    # === Model parameters === #
    hf_pretrained_model_name_or_path = zconf.attr(type=str, required=True)
    model_path = zconf.attr(type=str, required=True)
    model_config_path = zconf.attr(default=None, type=str)
    model_load_mode = zconf.attr(default="from_transformers", type=str)

    # === Running Setup === #
    do_train = zconf.attr(action="store_true")
    do_val = zconf.attr(action="store_true")
    do_save = zconf.attr(action="store_true")
    do_save_last = zconf.attr(action="store_true")
    do_save_best = zconf.attr(action="store_true")
    write_val_preds = zconf.attr(action="store_true")
    write_test_preds = zconf.attr(action="store_true")
    eval_every_steps = zconf.attr(type=int, default=0)
    min_train_steps = zconf.attr(type=int, default=0)# maple
    save_every_steps = zconf.attr(type=int, default=0)
    save_checkpoint_every_steps = zconf.attr(type=int, default=0)
    no_improvements_for_n_evals = zconf.attr(type=int, default=0)
    keep_checkpoint_when_done = zconf.attr(action="store_true")
    force_overwrite = zconf.attr(action="store_true")
    seed = zconf.attr(type=int, default=-1)

    # === Training Learning Parameters === #
    learning_rate = zconf.attr(default=1e-5, type=float)
    adam_epsilon = zconf.attr(default=1e-8, type=float)
    max_grad_norm = zconf.attr(default=1.0, type=float)
    optimizer_type = zconf.attr(default="adam", type=str)

    # Specialized config
    no_cuda = zconf.attr(action="store_true")
    fp16 = zconf.attr(action="store_true")
    fp16_opt_level = zconf.attr(default="O1", type=str)
    local_rank = zconf.attr(default=-1, type=int)
    server_ip = zconf.attr(default="", type=str)
    server_port = zconf.attr(default="", type=str)


@zconf.run_config
class ResumeConfiguration(zconf.RunConfig):
    checkpoint_path = zconf.attr(type=str)

def setup_runner(
    args: RunConfiguration,
    jiant_task_container: my_container_setup.JiantTaskContainer,
    quick_init_out,
    verbose: bool = True,
) -> my_jiant_runner2.JiantRunner:
    
    with distributed.only_first_process(local_rank=args.local_rank):
        # load the model
        jiant_model = jiant_model_setup.setup_jiant_model(
            hf_pretrained_model_name_or_path = args.hf_pretrained_model_name_or_path,
            model_config_path = args.model_config_path,
            task_dict = jiant_task_container.task_dict,
            taskmodels_config = jiant_task_container.taskmodels_config,
        )
        jiant_model_setup.delegate_load_from_path(
            jiant_model = jiant_model, weights_path = args.model_path, load_mode = args.model_load_mode
        )
        jiant_model.to(quick_init_out.device)

    user_mode = {
        e.split('=')[0]: e.split('=')[1] if len(e.split('=')) > 1 else None 
        for e in (args.user_mode[0].split(',') if type(args.user_mode) is not str else args.user_mode.split(','))
    }
    
    optimizer_scheduler = my_model_setup.create_optimizer(
        model = jiant_model, 
        learning_rate = args.learning_rate,
        t_total = jiant_task_container.global_train_config.max_steps,
        warmup_steps = jiant_task_container.global_train_config.warmup_steps,
        warmup_proportion = None,
        optimizer_type = args.optimizer_type,
        verbose = verbose,
    )

    jiant_model, optimizer = my_model_setup.raw_special_model_setup(
        model = jiant_model,
        optimizer = optimizer_scheduler.optimizer,
        fp16 = args.fp16,
        fp16_opt_level = args.fp16_opt_level,
        n_gpu = quick_init_out.n_gpu,
        local_rank = args.local_rank,
    )

    optimizer_scheduler.optimizer = optimizer
    
    rparams = my_jiant_runner2.RunnerParameters(
        local_rank = args.local_rank,
        n_gpu = quick_init_out.n_gpu,
        fp16 = args.fp16,
        max_grad_norm = args.max_grad_norm,  # 1.0
    )
    
    # create a runner
    runner = my_jiant_runner2.JiantRunner(
        jiant_task_container = jiant_task_container,
        jiant_model = jiant_model,
        optimizer_scheduler = optimizer_scheduler,
        device = quick_init_out.device,
        rparams = rparams,
        log_writer = quick_init_out.log_writer,
    )

    # create a sampler
    if int(user_mode['mcmc'])==1:
        runner.sampler = my_sampler.Sampler(
            jiant_model, float(user_mode['alp']), float(user_mode['bet']), 
            float(user_mode['lamb_init']), float(user_mode['Nexp']), float(user_mode['lr_others'])
        ).to(quick_init_out.device)
        runner.sampler.create_optimizer()
        runner.zero = copy.deepcopy(jiant_model)
    else:
        runner.sampler = my_sampler.Sampler(
            jiant_model, None, None, 1.0, 0, 0
        ).to(quick_init_out.device)
        #runner.zero = copy.deepcopy(jiant_model)  # debug purpose
    
    runner.user_mode = user_mode

    # create my logger
    runner.logger = my_utils.Logger.get(os.path.join(args.output_dir, "my_log.txt"))
    runner.output_dir = args.output_dir

    # load lambda stats
    runner.logger.info('Load lambda stats checkpoint from %s' % 
        (os.path.join(f"{user_mode['lamb_path']}"),))
    ckpt = torch.load(f=f"{user_mode['lamb_path']}", map_location=quick_init_out.device)
    runner.post_lamb_mean = copy.deepcopy(runner.sampler.lamb_)
    runner.post_lamb_mean.load_state_dict(state_dict=ckpt['mean'])

    runner.sp = sp = float(user_mode['splevel'])

    if True:

        vec = torch.nn.utils.parameters_to_vector(runner.post_lamb_mean)
        sort_vals, sort_idxs = vec.sort()
        idx = sort_idxs[-int(len(vec)*sp):]
        runner.logger.info(
            '**** Sparsity level = %s equals threshold lambda value = %.6f ****' % 
            (sp, sort_vals[-int(len(vec)*sp)]))
        runner.logger.info('Some suggesions:')
        strout = []
        for _ in list(np.linspace(0,0.01,41))[1:]:
            runner.logger.info(
                '    Sparsity level = %.3f corresponds to lambda threshold = %.6f' % 
                (_, sort_vals[-int(len(vec)*_)].item()))
            strout.append([_, sort_vals[-int(len(vec)*_)].item()])
        runner.logger.info(strout)
        runner.logger.info('----')
        strout = []
        for _ in list(np.linspace(0,0.1,81))[1:]:
            runner.logger.info(
                '    Sparsity level = %.3f corresponds to lambda threshold = %.6f' % 
                (_, sort_vals[-int(len(vec)*_)].item()))
            strout.append([_, sort_vals[-int(len(vec)*_)].item()])
        runner.logger.info(strout)
        runner.logger.info('----')
        strout = []
        for _ in list(np.linspace(0,1,1001))[1:]:
            runner.logger.info(
                '    Sparsity level = %.3f corresponds to lambda threshold = %.6f' % 
                (_, sort_vals[-int(len(vec)*_)].item()))
            strout.append([_, sort_vals[-int(len(vec)*_)].item()])
        runner.logger.info(strout)
        
        bs_mask = torch.ones_like(vec).bool()  # False = update, True = frozen
        bs_mask[idx] = False
        runner.bs_mask = [torch.zeros_like(p).bool() for p in runner.sampler.lamb_]
        torch.nn.utils.vector_to_parameters(bs_mask, runner.bs_mask)

        runner.logger.info('---- Selected weight stats: \n%s' % 
            (runner.sampler.print_lambda_sparsity(runner.jiant_model, runner.bs_mask),))

        runner.logger.info('---- Layer/module-wise sparsity patterns: \n%s' % 
            (runner.sampler.print_sparsity_patterns(runner.jiant_model, runner.bs_mask),))
    
    elif False:  # "layerwisely"
        runner.bs_mask = [torch.ones_like(p).bool() for p in runner.sampler.lamb_]
        for bs_mask, lamb in zip(runner.bs_mask, runner.post_lamb_mean):
            sort_vals, sort_idxs = lamb.flatten().sort()
            idx = sort_idxs[-int(lamb.numel()*sp):]
            bs_mask.flatten()[idx] = False
        
        runner.logger.info('---- Selected weight stats: \n%s' % 
            (runner.sampler.print_lambda_sparsity(runner.jiant_model, runner.bs_mask),))    
    return runner

def run_loop(args: RunConfiguration, checkpoint=None):
    is_resumed = checkpoint is not None
    
    quick_init_out = initialization.quick_init(args=args, verbose=True)
    quick_init_out.log_writer = my_zlog.ZLogger(
        os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y%m%d%H%m%S")), 
        overwrite = True)
    print(quick_init_out.n_gpu)

    with quick_init_out.log_writer.log_context():
        jiant_task_container = my_container_setup.create_jiant_task_container_from_json(
            jiant_task_container_config_path = args.jiant_task_container_config_path, 
            verbose = True)

        # setup jiant model, optimizer, and runner
        runner = setup_runner(
            args = args,
            jiant_task_container = jiant_task_container,
            quick_init_out = quick_init_out,
            verbose = True,
        )
        if is_resumed:
            runner.load_state(checkpoint["runner_state"])
            del checkpoint["runner_state"]

        checkpoint_saver = my_jiant_runner2.CheckpointSaver(
            metadata={"args": args.to_dict()},
            save_path=os.path.join(args.output_dir, "checkpoint.p"),
        )

        # training
        if args.do_train:
            
            # create a meta-runner
            metarunner = my_jiant_metarunner.JiantMetarunner(
                runner = runner,
                save_every_steps = args.save_every_steps,
                eval_every_steps = args.eval_every_steps,
                min_train_steps = args.min_train_steps,
                save_checkpoint_every_steps = args.save_checkpoint_every_steps,
                no_improvements_for_n_evals = args.no_improvements_for_n_evals,
                checkpoint_saver = checkpoint_saver,
                output_dir = args.output_dir,
                verbose = True,
                save_best_model = args.do_save or args.do_save_best,
                save_last_model = args.do_save or args.do_save_last,
                load_best_model = True,
                log_writer = quick_init_out.log_writer,
            )

            if is_resumed:
                metarunner.load_state(checkpoint["metarunner_state"])
                del checkpoint["metarunner_state"]
            metarunner.run_train_loop()

        # validation
        if args.do_val:
            val_results_dict = runner.run_val(
                task_name_list = runner.jiant_task_container.task_run_config.val_task_list,
                return_preds = args.write_val_preds,
            )
            
            my_writer.write_val_results(
                val_results_dict = val_results_dict,
                metrics_aggregator = runner.jiant_task_container.metrics_aggregator,
                output_dir = args.output_dir,
                verbose = True,
            )

            if args.write_val_preds:
                my_writer.write_preds(
                    eval_results_dict = val_results_dict,
                    path = os.path.join(args.output_dir, "val_preds.p"),
                )

        else:
            assert not args.write_val_preds

        # test
        if args.do_test:

            test_results_dict = runner.run_val(
                task_name_list = runner.jiant_task_container.task_run_config.test_task_list,
                return_preds = False,
                phase = "test"
            )
            
            my_writer.write_val_results(
                val_results_dict = test_results_dict,
                metrics_aggregator = runner.jiant_task_container.metrics_aggregator,
                output_dir = args.output_dir,
                verbose = True,
                result_file = "test_metrics.json"
            )
            
            train_results_dict = runner.run_val(
                task_name_list = runner.jiant_task_container.task_run_config.test_task_list,
                return_preds = False,
                phase = "train"
            )
            
            my_writer.write_val_results(
                val_results_dict = train_results_dict,
                metrics_aggregator = runner.jiant_task_container.metrics_aggregator,
                output_dir = args.output_dir,
                verbose = True,
                result_file = "train_metrics.json"
            )

        if args.write_test_preds:
            test_results_dict = runner.run_test(
                task_name_list = runner.jiant_task_container.task_run_config.test_task_list,
            )
            my_writer.write_preds(
                eval_results_dict = test_results_dict,
                path = os.path.join(args.output_dir, "test_preds.p"),
            )

    if (not args.keep_checkpoint_when_done
        and args.save_checkpoint_every_steps
        and os.path.exists(os.path.join(args.output_dir, "checkpoint.p"))
    ):
        os.remove(os.path.join(args.output_dir, "checkpoint.p"))

    py_io.write_file("DONE", os.path.join(args.output_dir, "done_file"))

def run_resume(args: ResumeConfiguration):
    resume(checkpoint_path=args.checkpoint_path)

def resume(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    args = RunConfiguration.from_dict(checkpoint["metadata"]["args"])
    run_loop(args=args, checkpoint=checkpoint)

def run_with_continue(cl_args):
    run_args = RunConfiguration.default_run_cli(cl_args=cl_args)
    if not run_args.force_overwrite and (
        os.path.exists(os.path.join(run_args.output_dir, "done_file"))
        or os.path.exists(os.path.join(run_args.output_dir, "val_metrics.json"))
    ):
        print("Already Done")
        return
    elif run_args.save_checkpoint_every_steps and os.path.exists(
        os.path.join(run_args.output_dir, "checkpoint.p")
    ):
        print("Resuming")
        resume(os.path.join(run_args.output_dir, "checkpoint.p"))
    else:
        print("Running from start")
        run_loop(args=run_args)


def main():
    mode, cl_args = zconf.get_mode_and_cl_args()
    if mode == "run":
        run_loop(RunConfiguration.default_run_cli(cl_args=cl_args))
    elif mode == "continue":
        run_resume(ResumeConfiguration.default_run_cli(cl_args=cl_args))
    elif mode == "run_with_continue":
        run_with_continue(cl_args=cl_args)
    else:
        raise zconf.ModeLookupError(mode)

if __name__ == "__main__":
    main()
