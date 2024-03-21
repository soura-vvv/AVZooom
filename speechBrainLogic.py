Method: __init__
    def __init__(  # noqa: C901
        self,
        modules=None,
        opt_class=None,
        hparams=None,
        run_opts=None,
        checkpointer=None,
        profiler=None,
    ):
        self.optimizers_dict = None
        self.opt_class = opt_class
        self.checkpointer = checkpointer
        self.profiler = profiler

        for arg, default in run_opt_defaults.items():
            if run_opts is not None and arg in run_opts:
                if hparams is not None and arg in hparams:
                    logger.info(
                        "Info: "
                        + arg
                        + " arg overridden by command line input to: "
                        + str(run_opts[arg])
                    )
                setattr(self, arg, run_opts[arg])
            else:
                # If any arg from run_opt_defaults exist in hparams and
                # not in command line args "run_opts"
                if hparams is not None and arg in hparams:
                    logger.info(
                        "Info: " + arg + " arg from hparam file is used"
                    )
                    setattr(self, arg, hparams[arg])
                else:
                    setattr(self, arg, default)

        # Check Python version
        if not (
            sys.version_info.major == PYTHON_VERSION_MAJOR
            and sys.version_info.minor >= PYTHON_VERSION_MINOR
        ):
            logger.warning(
                "Detected Python "
                + str(sys.version_info.major)
                + "."
                + str(sys.version_info.minor)
                + ". We suggest using SpeechBrain with Python >="
                + str(PYTHON_VERSION_MAJOR)
                + "."
                + str(PYTHON_VERSION_MINOR)
            )

        # Assume `torchrun` was used if `RANK` and `LOCAL_RANK` are set
        self.distributed_launch = (
            os.environ.get("RANK") is not None
            and os.environ.get("LOCAL_RANK") is not None
        )

        if self.data_parallel_backend and self.distributed_launch:
            raise ValueError(
                "To use data_parallel backend, start your script with:\n\t"
                "python experiment.py hyperparams.yaml "
                "--data_parallel_backend=True\n"
                "To use DDP backend, start your script with:\n\t"
                "torchrun [args] experiment.py hyperparams.yaml"
            )

        if self.ckpt_interval_minutes > 0 and self.ckpt_interval_steps > 0:
            sys.exit(
                "The options `ckpt_interval_minutes` and `ckpt_interval_steps` "
                "are mutually exclusive. "
                "Please keep only one active per experiment run."
            )

        # Switch to the right context
        if self.device == "cuda":
            torch.cuda.set_device(0)
        elif "cuda" in self.device:
            torch.cuda.set_device(int(self.device[-1]))

        # Put modules on the right device, accessible with dot notation
        self.modules = torch.nn.ModuleDict(modules).to(self.device)

        # The next line ensures that both tensors marked as parameters and standard tensors,
        # such as those used in InputNormalization, are placed on the right device.
        for module in self.modules:
            if hasattr(self.modules[module], "to"):
                self.modules[module] = self.modules[module].to(self.device)

        # Make hyperparams available with dot notation too
        if hparams is not None:
            self.hparams = SimpleNamespace(**hparams)

        # Checkpointer should point at a temporary directory in debug mode
        if (
            self.debug
            and not self.debug_persistently
            and self.checkpointer is not None
            and hasattr(self.checkpointer, "checkpoints_dir")
        ):
            tempdir = tempfile.TemporaryDirectory()
            logger.info(
                "Since debug mode is active, switching checkpointer "
                f"output to temporary directory: {tempdir.name}"
            )
            self.checkpointer.checkpoints_dir = pathlib.Path(tempdir.name)

            # Keep reference to tempdir as long as checkpointer exists
            self.checkpointer.tempdir = tempdir

        # Sampler should be handled by `make_dataloader`
        # or if you provide a DataLoader directly, you can set
        # this.train_sampler = your_sampler
        # to have your_sampler.set_epoch() called on each epoch.
        self.train_sampler = None

        if self.auto_mix_prec:
            logger.warning(
                "The option `--auto_mix_prec` is deprecated and will be removed in the future. "
                "Please use `--precision=fp16` instead."
            )
            self.precision = "fp16"

        if self.bfloat16_mix_prec:
            logger.warning(
                "The option `--bfloat16_mix_prec` is deprecated and will be removed in the future. "
                "Please use `--precision=bf16` instead."
            )
            self.precision = "bf16"

        if self.device == "cpu" and self.precision == "fp16":
            raise ValueError(
                "The option `--precision` is enabled with the value "
                "fp16. This option is not yet supported on CPU. "
                "Please use `--precision=bf16` instead to get "
                "mixed precision on CPU."
            )

        gradscaler_enabled = self.precision == "fp16" and "cuda" in self.device
        if self.skip_nonfinite_grads and gradscaler_enabled:
            logger.warning(
                "The option `skip_nonfinite_grads` will be ignored "
                "because GradScaler is enabled and will automatically "
                "skip nonfinite gradients."
            )

        logger.info(
            f"Gradscaler enabled: {gradscaler_enabled}. Using precision: {self.precision}."
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=gradscaler_enabled)

        self.use_amp = False
        if self.device == "cpu" and self.precision == "bf16":
            self.use_amp = True
        elif "cuda" in self.device and self.precision in ["fp16", "bf16"]:
            self.use_amp = True

        if self.use_amp and self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "scaler", self.scaler, optional_load=True
            )

        # List parameter count for the user
        total_params = sum(
            p.numel() for p in self.modules.parameters() if p.requires_grad
        )
        if total_params > 0:
            clsname = self.__class__.__name__
            fmt_num = sb.utils.logger.format_order_of_magnitude(total_params)
            logger.info(f"{fmt_num} trainable parameters in {clsname}")

        if self.distributed_launch:
            self.rank = int(os.environ["RANK"])
            if not torch.distributed.is_initialized():
                if self.rank > 0:
                    raise ValueError(
                        " ================ WARNING ==============="
                        "Please add sb.ddp_init_group() into your exp.py"
                        "To use DDP backend, start your script with:\n\t"
                        "torchrun [args] experiment.py hyperparams.yaml"
                    )
                else:
                    logger.warning(
                        "To use DDP, please add "
                        "sb.utils.distributed.ddp_init_group() into your exp.py"
                    )
                    logger.info(
                        "Only the main process is alive, "
                        "all other subprocess were killed."
                    )

        # Prepare iterating variables
        self.avg_train_loss = 0.0
        self.step = 0
        self.optimizer_step = 0

        # Add this class to the checkpointer for intra-epoch checkpoints
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("brain", self)

        # Force default color for tqdm progrressbar
        if not self.tqdm_colored_bar:
            self.tqdm_barcolor = dict.fromkeys(self.tqdm_barcolor, "")



Method: _compile
    def _compile(self):
        """Compile requested modules with either JIT or TorchInductor."""
        compile_available = hasattr(torch, "compile")

        if not compile_available and self.compile_module_keys is not None:
            raise ValueError(
                "'compile_module_keys' specified, but this install of PyTorch "
                "seems to be too old to support it."
            )
        # Modules to compile with torch.compile
        compile_module_keys = set()
        if self.compile:
            if self.compile_module_keys is None:
                compile_module_keys = set(self.modules)
            else:
                compile_module_keys = set(self.compile_module_keys)
                logger.warning(
                    "--compile and --compile_module_keys are both specified. "
                    "Only modules specified in --compile_module_keys will be compiled."
                )

        # Modules to compile with jit
        jit_module_keys = set()
        if self.jit:
            if self.jit_module_keys is None:
                jit_module_keys = set(self.modules)
            else:
                jit_module_keys = set(self.jit_module_keys)
                logger.warning(
                    "--jit and --jit_module_keys are both specified. "
                    "Only modules specified in --jit_module_keys will be compiled."
                )

        # find missing keys
        for name in compile_module_keys | jit_module_keys:
            if name not in self.modules:
                raise ValueError(
                    f"module {name} is not defined in your hparams file."
                )

        # try 'torch.compile', remove successful compiles from JIT list
        for name in compile_module_keys:
            try:
                module = torch.compile(
                    self.modules[name],
                    mode=self.compile_mode,
                    fullgraph=self.compile_using_fullgraph,
                    dynamic=self.compile_using_dynamic_shape_tracing,
                )
            except Exception as e:
                logger.warning(
                    f"'{name}' in 'compile_module_keys' failed to compile "
                    f"and will be skipped (may fallback onto JIT, if "
                    f"specified): {e}"
                )
                continue

            self.modules[name] = module.to(self.device)
            jit_module_keys.discard(name)

        for name in jit_module_keys:
            module = torch.jit.script(self.modules[name])
            self.modules[name] = module.to(self.device)



Method: _fit_train
    def _fit_train(self, train_set, epoch, enable):
        # Training stage
        self.on_stage_start(Stage.TRAIN, epoch)
        self.modules.train()
        self.zero_grad()

        # Reset nonfinite count to 0 each epoch
        self.nonfinite_count = 0

        if self.train_sampler is not None and hasattr(
            self.train_sampler, "set_epoch"
        ):
            self.train_sampler.set_epoch(epoch)

        # Time since last intra-epoch checkpoint
        last_ckpt_time = time.time()
        steps_since_ckpt = 0
        with tqdm(
            train_set,
            initial=self.step,
            dynamic_ncols=True,
            disable=not enable,
            colour=self.tqdm_barcolor["train"],
        ) as t:
            for batch in t:
                if self._optimizer_step_limit_exceeded:
                    logger.info("Train iteration limit exceeded")
                    break
                self.step += 1
                steps_since_ckpt += 1
                loss = self.fit_batch(batch)
                self.avg_train_loss = self.update_average(
                    loss, self.avg_train_loss
                )
                t.set_postfix(train_loss=self.avg_train_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

                if self._should_save_intra_epoch_ckpt(
                    last_ckpt_time, steps_since_ckpt
                ):
                    # Checkpointer class will handle running this on main only
                    self._save_intra_epoch_ckpt()
                    last_ckpt_time = time.time()
                    steps_since_ckpt = 0

        # Run train "on_stage_end" on all processes
        self.zero_grad(set_to_none=True)  # flush gradients
        self.on_stage_end(Stage.TRAIN, self.avg_train_loss, epoch)
        self.avg_train_loss = 0.0
        self.step = 0



Method: _fit_valid
    def _fit_valid(self, valid_set, epoch, enable):
        # Validation stage
        if valid_set is not None:
            self.on_stage_start(Stage.VALID, epoch)
            self.modules.eval()
            avg_valid_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(
                    valid_set,
                    dynamic_ncols=True,
                    disable=not enable,
                    colour=self.tqdm_barcolor["valid"],
                ):
                    self.step += 1
                    loss = self.evaluate_batch(batch, stage=Stage.VALID)
                    avg_valid_loss = self.update_average(loss, avg_valid_loss)

                    # Profile only if desired (steps allow the profiler to know when all is warmed up)
                    if self.profiler is not None:
                        if self.profiler.record_steps:
                            self.profiler.step()

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                self.step = 0
                self.on_stage_end(Stage.VALID, avg_valid_loss, epoch)



Method: _recover
    @sb.utils.checkpoints.mark_as_loader
    def _recover(self, path, end_of_epoch):
        del end_of_epoch
        with open(path) as f:
            save_dict = yaml.safe_load(f)
        self.step = save_dict["step"]
        self.avg_train_loss = save_dict["avg_train_loss"]
        # Ensure compatibility with checkpoints from before optimizer_step:
        if "optimizer_step" not in save_dict:
            clsname = self.__class__.__name__
            MSG = f"'optimizer_step' not found in {clsname} checkpoint."
            MSG += " Using the saved 'step' value (BACKWARDS COMPATIBILITY)"
            warnings.warn(MSG)
            self.optimizer_step = self.step
        else:
            self.optimizer_step = save_dict["optimizer_step"]



Method: _save
    @sb.utils.checkpoints.mark_as_saver
    def _save(self, path):
        save_dict = {
            "step": self.step,
            "avg_train_loss": self.avg_train_loss,
            "optimizer_step": self.optimizer_step,
        }
        with open(path, "w") as w:
            w.write(yaml.dump(save_dict))



Method: _save_intra_epoch_ckpt
    def _save_intra_epoch_ckpt(self):
        """Saves a CKPT with specific intra-epoch flag."""
        self.checkpointer.save_and_keep_only(
            end_of_epoch=False,
            num_to_keep=1,
            ckpt_predicate=lambda c: INTRA_EPOCH_CKPT_FLAG in c.meta,
            meta={INTRA_EPOCH_CKPT_FLAG: True},
            verbosity=logging.DEBUG,
        )



Method: _should_save_intra_epoch_ckpt
    def _should_save_intra_epoch_ckpt(self, last_ckpt_time, steps_since_ckpt):
        """Determines if an intra-epoch checkpoint should be saved.

        Returns True if there's a checkpointer and time or steps has exceeded limit.
        """
        if self.checkpointer is None:
            return False

        # Return early if mid-epoch checkpoints are disabled to avoid sync
        if self.ckpt_interval_minutes <= 0 and self.ckpt_interval_steps <= 0:
            return False

        # Check if we've run for the requested amount of time
        elapsed_minutes = (time.time() - last_ckpt_time) / 60.0
        decision = 0 < self.ckpt_interval_minutes < elapsed_minutes

        # Save after requested # of steps
        decision = decision or 0 < self.ckpt_interval_steps <= steps_since_ckpt

        # If the program is not distributed, just return
        if not torch.distributed.is_initialized():
            return decision

        # Otherwise, broadcast decision to all processes from main (rank 0)
        # This solves synchronization issues where main gets a different
        # timing result than the other processes.
        else:
            broadcast_list = [decision]
            torch.distributed.broadcast_object_list(broadcast_list, src=0)
            return broadcast_list[0]



Method: _train_loader_specifics
    def _train_loader_specifics(self, dataset, loader_kwargs):
        sampler = loader_kwargs.get("sampler", None)
        # Shuffling should really only matter for the train stage. Shuffling
        # will also lead to more padding in batches if the order was otherwise
        # sorted by length.
        shuffle = loader_kwargs.get("shuffle", False)
        if shuffle and not self.distributed_launch:
            if sampler is not None:
                raise ValueError(
                    "Cannot specify both shuffle=True"
                    "and a sampler in loader_kwargs"
                )
            sampler = ReproducibleRandomSampler(dataset)
            self.train_sampler = sampler
            loader_kwargs["sampler"] = self.train_sampler
            # Delete the shuffle flag, since you cannot specify both a sampler and
            # shuffling:
            del loader_kwargs["shuffle"]

        # Possibly make a DistributedSampler or a wrapper for some other sampler
        if self.distributed_launch and not isinstance(dataset, IterableDataset):
            # sort or not
            if hasattr(self.hparams, "sorting"):
                shuffle_ddp = (
                    self.hparams.sorting == "random"
                )  # False if 'ascending' or 'descending'
            else:
                shuffle_ddp = True

            drop_last = loader_kwargs.get("drop_last", False)
            # num_replicas arg is equal to world_size
            # and retrieved automatically within
            # DistributedSampler obj.
            if sampler is not None:
                self.train_sampler = DistributedSamplerWrapper(
                    sampler,
                    rank=self.rank,
                    drop_last=drop_last,
                    shuffle=shuffle,
                )

                # with DistributedSamplerWrapper, one must disable shuffling for dataloader
                loader_kwargs["shuffle"] = False
                loader_kwargs["sampler"] = self.train_sampler
            elif loader_kwargs.get("batch_sampler") is None:
                # no sampler and batch-sampler
                self.train_sampler = DistributedSampler(
                    dataset,
                    rank=self.rank,
                    shuffle=shuffle_ddp,
                    drop_last=drop_last,
                )

                # with DistributedSamplerWrapper, one must disable shuffling for dataloader
                loader_kwargs["shuffle"] = False
                loader_kwargs["sampler"] = self.train_sampler
            else:  # batch_sampler was specified
                self.train_sampler = DistributedSamplerWrapper(
                    loader_kwargs.get("batch_sampler", None),
                    rank=self.rank,
                    shuffle=shuffle_ddp,
                )
                loader_kwargs["batch_sampler"] = self.train_sampler
        elif self.distributed_launch and isinstance(dataset, IterableDataset):
            logger.warning(
                "Cannot automatically solve distributed sampling "
                "for IterableDataset."
            )
        return loader_kwargs



Method: _wrap_distributed
    def _wrap_distributed(self):
        """Wrap modules with distributed wrapper when requested."""
        if not self.distributed_launch and not self.data_parallel_backend:
            return
        elif self.distributed_launch:
            for name, module in self.modules.items():
                if any(p.requires_grad for p in module.parameters()):
                    module = SyncBatchNorm.convert_sync_batchnorm(module)
                    if self.distributed_backend == "gloo":
                        module = DDP(
                            module,
                            device_ids=None,
                            find_unused_parameters=self.find_unused_parameters,
                        )
                    else:
                        module = DDP(
                            module,
                            device_ids=[self.device],
                            find_unused_parameters=self.find_unused_parameters,
                        )
                    self.modules[name] = module
        else:
            # data_parallel_backend
            for name, module in self.modules.items():
                if any(p.requires_grad for p in module.parameters()):
                    module = DP(module)
                    self.modules[name] = module



Method: check_gradients
    def check_gradients(self):
        """ Checks if the gradients are finite. If not, it will emit a warning and set them to zero."""
        for param in self.modules.parameters():
            if param.requires_grad and param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    param.grad = None
                    logger.warning(
                        f"Gradients {param.name} contain NaN or Inf. Setting to None."
                    )



Method: check_loss_isfinite
    def check_loss_isfinite(self, loss):
        """Check if the loss is finite.

        If the loss is not finite, log a helpful message and increment the `nonfinite_count`.
        If the `nonfinite_count` exceeds the `--nonfinite_patience` threshold, stop the training
        and raise an error.

        This check is particularly useful when the loss becomes NaN or inf, while the
        parameters and gradients remain finite. It helps prevent getting stuck in an
        infinite loop during training.

        Arguments
        ---------
        loss : tensor
            The loss tensor after ``backward()`` has been called but
            before the optimizers ``step()``.
        """
        if not torch.isfinite(loss):
            self.nonfinite_count += 1

            # Check if patience is exhausted
            if self.nonfinite_count > self.nonfinite_patience:
                raise ValueError(
                    "Loss is not finite and patience is exhausted. "
                    "To debug, wrap `fit()` with "
                    "autograd's `detect_anomaly()`, e.g.\n\nwith "
                    "torch.autograd.detect_anomaly():\n\tbrain.fit(...)"
                )
            else:
                logger.warning("Patience not yet exhausted.")



Method: compute_feats
    def compute_feats(self, wavs):
        """Returns corresponding log-spectral features of the input waveforms.

        Arguments
        ---------
        wavs : torch.Tensor
            The batch of waveforms to convert to log-spectral features.
        """

        # Log-spectral features
        feats = self.hparams.compute_STFT(wavs)
        feats = sb.processing.features.spectral_magnitude(feats, power=0.5)

        # Log1p reduces the emphasis on small differences
        feats = torch.log1p(feats)

        #Sourav Change
        print(feats.size())
        temp_zeros=torch.zeros(feats.size(dim=0),feats.size(dim=1),2).to(device)
        #print("Temp_Zeros Size:")
        #print(temp_zeros.size())
        #print("Feats Size:")
        #print(feats.size())
        feats=torch.cat((feats,temp_zeros),2).to(device)
        #print("Post Feats Size:")
        #print(feats.size())
        #Sourav Change End
        return feats



Method: compute_forward
    def compute_forward(self, batch, stage):
        """Apply masking to convert from noisy waveforms to enhanced signals.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : dict
            A dictionary with keys {"spec", "wav"} with predicted features.
        """

        # We first move the batch to the appropriate device, and
        # compute the features necessary for masking.
        batch = batch.to(self.device)
        self.clean_wavs, self.lens = batch.clean_sig

        noisy_wavs, self.lens = self.hparams.wav_augment(
            self.clean_wavs, self.lens
        )

        noisy_feats = self.compute_feats(noisy_wavs)

        # Masking is done here with the "signal approximation (SA)" algorithm.
        # The masked input is compared directly with clean speech targets.
        
        #Actual Training
        mask = self.modules.model(noisy_feats)
        
        #make mask 259
        temp_zeros=torch.zeros(mask.size(dim=0),mask.size(dim=1),2).to(device)
        mask=torch.cat((mask,temp_zeros),2).to(device)
        
        #noisy_feats_chopped=torch.split(noisy_feats,257,dim=2)
        #print("New Dimension:")
        #print(noisy_feats)
        #print("Mask Time:")
        #print(mask.size())
        
        predict_spec = torch.mul(mask, noisy_feats)
        
        #predict_spec=torch.mul(mask,noisy_feats)
        # Also return predicted wav, for evaluation. Note that this could
        # also be used for a time-domain loss term.
        predict_spec_chopped=torch.split(predict_spec,257,dim=2)
        predict_wav = self.hparams.resynth(
            torch.expm1(predict_spec_chopped[0]), noisy_wavs
        )
        #predict_wav=0

        # Return a dictionary so we don't have to remember the order
        return {"spec": predict_spec, "wav": predict_wav}



Method: compute_objectives
    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : dict
            The output dict from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        # Prepare clean targets for comparison
        clean_spec = self.compute_feats(self.clean_wavs)

        # Directly compare the masked spectrograms with the clean targets
        loss = sb.nnet.losses.mse_loss(
            predictions["spec"], clean_spec, self.lens
        )

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.id,
            predictions["spec"],
            clean_spec,
            self.lens,
            reduction="batch",
        )

        # Some evaluations are slower, and we only want to perform them
        # on the validation set.
        if stage != sb.Stage.TRAIN:

            # Evaluate speech intelligibility as an additional metric
            self.stoi_metric.append(
                batch.id,
                predictions["wav"],
                self.clean_wavs,
                self.lens,
                reduction="batch",
            )

        return loss



Method: evaluate
    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """Iterate test_set and evaluate brain performance. By default, loads
        the best-performing checkpoint (as recorded using the checkpointer).

        Arguments
        ---------
        test_set : Dataset, DataLoader
            If a DataLoader is given, it is iterated directly. Otherwise passed
            to ``self.make_dataloader()``.
        max_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        min_key : str
            Key to use for finding best checkpoint, passed to
            ``on_evaluate_start()``.
        progressbar : bool
            Whether to display the progress in a progressbar.
        test_loader_kwargs : dict
            Kwargs passed to ``make_dataloader()`` if ``test_set`` is not a
            DataLoader. NOTE: ``loader_kwargs["ckpt_prefix"]`` gets
            automatically overwritten to ``None`` (so that the test DataLoader
            is not added to the checkpointer).

        Returns
        -------
        average test loss
        """
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                test_set,
                dynamic_ncols=True,
                disable=not progressbar,
                colour=self.tqdm_barcolor["test"],
            ):
                self.step += 1
                loss = self.evaluate_batch(batch, stage=Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            self.on_stage_end(Stage.TEST, avg_test_loss, None)
        self.step = 0
        return avg_test_loss



Method: evaluate_batch
    @torch.no_grad()
    def evaluate_batch(self, batch, stage):
        """Evaluate one batch, override for different procedure than train.

        The default implementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """

        out = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        return loss.detach().cpu()



Method: fit
    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``

        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.

        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """
        if self.test_only:
            logger.info(
                "Test only mode, skipping training and validation stages."
            )
            return

        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
            or isinstance(valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        # Iterate epochs
        for epoch in epoch_counter:
            self._fit_train(train_set=train_set, epoch=epoch, enable=enable)
            self._fit_valid(valid_set=valid_set, epoch=epoch, enable=enable)

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break



Method: fit_batch
    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``
        * ``optimizers_step()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        amp = AMPConfig.from_name(self.precision)
        should_step = (self.step % self.grad_accumulation_factor) == 0

        with self.no_sync(not should_step):
            if self.use_amp:
                with torch.autocast(
                    dtype=amp.dtype, device_type=torch.device(self.device).type,
                ):
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                    loss = self.compute_objectives(
                        outputs, batch, sb.Stage.TRAIN
                    )
            else:
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            scaled_loss = self.scaler.scale(
                loss / self.grad_accumulation_factor
            )
            self.check_loss_isfinite(scaled_loss)
            scaled_loss.backward()

        if should_step:
            self.optimizers_step()

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()



Method: freeze_optimizers
    def freeze_optimizers(self, optimizers):
        """By default, this method returns the passed optimizers.
        Override this method if you want to freeze some optimizers
        during training. To do so, return a of active optimizers.
        """
        return optimizers



Method: infer
    def infer(self,model, dataloader):
        model.eval()  # Set the model to evaluation mode
        print(dir(self))
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                
                # We first move the batch to the appropriate device, and
                # compute the features necessary for masking.
                print(batch)
                #batch = batch.to(self.device)
                self.clean_wavs= batch['clean_sig']
                self.lens=len(self.clean_wavs)
                #noisy_wavs, self.lens = self.hparams.wav_augment(
                #    self.clean_wavs, self.lens
                #)
                noisy_wavs=self.clean_wavs
                noisy_feats = self.compute_feats(noisy_wavs).to(device)

                # Masking is done here with the "signal approximation (SA)" algorithm.
                # The masked input is compared directly with clean speech targets.
        
                #Actual Going through the model
                mask = self.modules.model(noisy_feats)
        
                #make mask 259
                temp_zeros=torch.zeros(mask.size(dim=0),mask.size(dim=1),2).to(device)
                mask=torch.cat((mask,temp_zeros),2).to(device)
        
                #noisy_feats_chopped=torch.split(noisy_feats,257,dim=2)
                #print("New Dimension:")
                #print(noisy_feats)
                #print("Mask Time:")
                #print(mask.size())
                predict_spec = torch.mul(mask, noisy_feats)
                
                predict_spec_chopped=torch.split(predict_spec,257,dim=2)
                predict_wav = self.hparams.resynth(
                    torch.expm1(predict_spec_chopped[0]), noisy_wavs
                )
                predictions.append(predict_wav)
        return predictions



Method: init_optimizers
    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).

        The default implementation of this method depends on an optimizer
        class being passed at initialization that takes only a list
        of parameters (e.g., a lambda or a partial function definition).
        This creates a single optimizer that optimizes all trainable params.

        Override this class if there are multiple optimizers.
        """

        all_params = self.modules.parameters()

        if self.opt_class is not None:
            if self.remove_vector_weight_decay:
                all_params = rm_vector_weight_decay(self.modules)

            self.optimizer = self.opt_class(all_params)

            self.optimizers_dict = {"opt_class": self.optimizer}

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)



Method: make_dataloader
    def make_dataloader(
        self, dataset, stage, ckpt_prefix="dataloader-", **loader_kwargs
    ):
        """Creates DataLoaders for Datasets.

        This is used by ``fit()`` and ``evaluate()`` if they just receive
        Datasets.

        Alternatively, this can be called from outside the Brain subclass.
        In that case, the DataLoader should be passed to ``fit()`` in place
        of the dataset.

        The Stage.TRAIN DataLoader is handled specially. It has extra args for
        shuffle and drop_last. In DDP a DistributedSampler is created (unless
        the dataset is an IterableDataset).

        NOTE
        ----
        Some important DataLoader arguments are passed via **loader_kwargs,
        e.g., batch_size, num_workers, pin_memory.

        NOTE
        ----
        By default, ``evaluate()`` specifies ckpt_prefix=None to stop the test
        DataLoader being added to the checkpointer. If you need to add a
        recoverable after saving checkpoints (e.g., at test time, after
        checkpointing the training), and still be able to recover reasonably,
        you should probably specify ``allow_partial_load=True``.

        Arguments
        ---------
        dataset : Dataset
            A set of data to use to create data loader. If the Dataset is a
            DynamicItemDataset, PaddedBatch is used as the default collate_fn,
            unless specified in loader_kwargs.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST
        ckpt_prefix : str, None
            Prefix to use for SaveableDataLoader Checkpoint name. The Stage
            name is added to this to create the full key. Set to None to not
            save the DataLoader.
        **loader_kwargs : dict
            Additional keyword arguments to the DataLoader.
            E.g., batch_size, num_workers, pin_memory.
        """
        # TRAIN stage is handled specially.
        if stage == sb.Stage.TRAIN:
            loader_kwargs = self._train_loader_specifics(dataset, loader_kwargs)
        # This commented-out code block is useful when one can ensure
        # metric reporting is DDP-valid for VALID & EVAL datasets.
        # elif self.distributed_launch:
        #     loader_kwargs = sb.dataio.dataloader.distributed_loader_specifics(
        #         self.distributed_launch, self.rank, dataset, loader_kwargs
        #     )
        dataloader = sb.dataio.dataloader.make_dataloader(
            dataset, **loader_kwargs
        )

        if (
            self.checkpointer is not None
            and ckpt_prefix is not None
            and (
                isinstance(dataloader, SaveableDataLoader)
                or isinstance(dataloader, LoopedLoader)
            )
        ):
            ckpt_key = ckpt_prefix + stage.name
            self.checkpointer.add_recoverable(ckpt_key, dataloader)
        return dataloader



Method: no_sync
    @contextmanager
    def no_sync(self, use=True):
        """Copies pytorch's implementation for doing no_sync across all modules.

        Explanation: nn.module.no_sync() is a context manager for when one does
        not want to sync gradients, which happens when using both DDP and gradient accumulation.
        Speechbrain brain's class can contain multiple modules and calling no_sync on these
        individually would be very awkward, therefore this contextmanager exists.

        Arguments
        ---------
        use : bool
            If set to `False` will still sync gradients, useful to make behaviour togglable.
        """
        if use:
            old_values_list = []
            for module in self.modules.values():
                if not hasattr(module, "require_backward_grad_sync"):
                    # if not using DDP
                    continue
                old_values_list.append(module.require_backward_grad_sync)
                module.require_backward_grad_sync = False
            yield
            i = 0
            for module in self.modules.values():
                if not hasattr(module, "require_backward_grad_sync"):
                    continue
                module.require_backward_grad_sync = old_values_list[i]
                i += 1
        else:
            yield



Method: on_evaluate_start
    def on_evaluate_start(self, max_key=None, min_key=None):
        """Gets called at the beginning of ``evaluate()``

        Default implementation loads the best-performing checkpoint for
        evaluation, based on stored metrics.

        Arguments
        ---------
        max_key : str
            Key to use for finding best checkpoint (higher is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        min_key : str
            Key to use for finding best checkpoint (lower is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        """

        # Recover best checkpoint for evaluation
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                max_key=max_key, min_key=min_key,
            )



Method: on_fit_batch_end
    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Called after ``fit_batch()``.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        outputs : list or dictionary of torch.Tensors
            Returned value of compute_forward().
        loss : torch.Tensor
            Returned value of compute_objectives().
        should_step : boolean
            Whether optimizer.step() was called or not.
        """
        pass



Method: on_fit_batch_start
    def on_fit_batch_start(self, batch, should_step):
        """Called at the beginning of ``fit_batch()``.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        should_step : boolean
            Whether optimizer.step() was called or not.
        """
        pass



Method: on_fit_start
    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit/compiled modules
        # cannot be pickled.
        self._compile()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible()



Method: on_stage_end
    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "stoi": -self.stoi_metric.summarize("average"),
            }

        # At the end of validation, we can write stats and checkpoints
        if stage == sb.Stage.VALID:
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            # unless they have the current best STOI score.
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["stoi"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )



Method: on_stage_start
    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.mse_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.stoi_metric = sb.utils.metric_stats.MetricStats(
                metric=sb.nnet.loss.stoi_loss.stoi_loss
            )



Method: optimizers_step
    def optimizers_step(self):
        """Performs a step of gradient descent on the optimizers. This method is called every
        ``grad_accumulation_factor`` steps."""
        # 1. get the valid optimizers, i.e., the ones that are not frozen during this step
        if self.optimizers_dict is not None:
            valid_optimizers = self.freeze_optimizers(self.optimizers_dict)
        elif self.opt_class is not None:
            # if valid_optimizers is not defined which could happen if a user is using an old
            # init_optimizers() method, then we assume that the only valid optimizer is
            # self.optimizer (which is the default behavior).
            valid_optimizers = {"optimizer": self.optimizer}
        else:
            # Note: in some cases you might want to only compute gradients statistics and
            # you do not need to call the optimizers.step() method. In this case, you can
            # simply return from this method and skip the rest of the code.
            return

        # 2. unscale the gradients of the valid optimizers
        for opt in valid_optimizers.values():
            self.scaler.unscale_(opt)

        # 3. clip gradients
        # We are clipping this way because clipping on self.modules.parameters()
        # can leads to NaN/Inf gradients norm as doing the concatenation
        # of all parameters in a single vector can lead to overflow/underflow.
        for opt in valid_optimizers.values():
            torch.nn.utils.clip_grad_norm_(
                opt.param_groups[0]["params"], self.max_grad_norm
            )

        # Note: no need to activate this flag if you are in fp16
        # since GradScaler is automatically handling the nonfinite gradients
        if not self.scaler.is_enabled() and self.skip_nonfinite_grads:
            self.check_gradients()

        # 4. step the valid optimizers
        # If the scaler is disable, it simply calls optimizer.step()
        for opt in valid_optimizers.values():
            self.scaler.step(opt)

        self.scaler.update()

        for opt in valid_optimizers.values():
            opt.zero_grad(set_to_none=True)

        self.optimizer_step += 1



Method: update_average
    def update_average(self, loss, avg_loss):
        """Update running average of the loss.

        Arguments
        ---------
        loss : torch.tensor
            detached loss, a single float value.
        avg_loss : float
            current running average.

        Returns
        -------
        avg_loss : float
            The average loss.
        """
        if torch.isfinite(loss):
            avg_loss -= avg_loss / self.step
            avg_loss += float(loss) / self.step
        return avg_loss



Method: zero_grad
    def zero_grad(self, set_to_none=False):
        """Sets the gradients of all optimized ``torch.Tensor``s to zero
        if ``set_to_none=False`` (default) or to None otherwise.

        Setting gradients to None should save the memory, e.g.
        during ``evaluate()`` and thus larger batch might be used.
        """
        if self.optimizers_dict is not None:
            for opt in self.freeze_optimizers(self.optimizers_dict).values():
                opt.zero_grad(set_to_none=set_to_none)
        elif self.opt_class is not None:
            self.optimizer.zero_grad(set_to_none=set_to_none)


