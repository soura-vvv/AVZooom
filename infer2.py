
#!/usr/bin/env/python3
"""Recipe for training a speech enhancement system with spectral masking.

To run this recipe, do the following:
> python train.py train.yaml --data_folder /path/to/save/mini_librispeech

To read the code, first scroll to the bottom to see the "main" code.
This gives a high-level overview of what is going on, while the
Brain class definition provides the details of what happens
for each batch during training.

The first time you run it, this script should automatically download
and prepare the Mini Librispeech dataset for computation. Noise and
reverberation are automatically added to each sample from OpenRIR.

Authors
 * Szu-Wei Fu 2020
 * Chien-Feng Liao 2020
 * Peter Plantinga 2021
"""
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from mini_librispeech_prepare import prepare_mini_librispeech
import inspect
from tqdm import tqdm


# Brain class for speech enhancement training
class SEBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""




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
        print("Batchsdhosijd---------------------")
        print(batch)
        batch = batch.to(self.device)
        #self.clean_wavs, self.lens = batch.clean_sig
        self.clean_wavs=batch["clean_sig"]
        #noisy_wavs, self.lens = self.hparams.wav_augment(
        #    self.clean_wavs, self.lens
        #)
        noisy_wavs=self.clean_wavs
        noisy_feats = self.compute_feats(noisy_wavs)
        print("Noisy Feats Size")
        print(noisy_feats.size())
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

        #if not (
        #    isinstance(test_set, DataLoader)
        #    or isinstance(test_set, LoopedLoader)
        #):
        #    test_loader_kwargs["ckpt_prefix"] = None
        #    test_set = self.make_dataloader(
        #        test_set, Stage.TEST, **test_loader_kwargs
        #    )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        #self.on_stage_start(Stage.TEST, epoch=None)
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
                loss,out = self.evaluate_batch(batch, "")
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
        return avg_test_loss,out

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
        return loss.detach().cpu(),out
    # Define the inference function
    def infer2(self,test_set, max_key=None,min_key=None,progressbar=None,test_loader_kwargs={}):
        if progressbar is None:
            progressbar = not self.noprogressbar
            
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.modules.eval()
        with torch.no_grad():
            for batch in tqdm(
                test_set,
                dynamic_ncols=True,
                disable=not progressbar,
                colour=self.tqdm_barcolor["test"],
            ):
                self.step += 1
                loss,out = self.evaluate_batch(batch, stage="")
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            #self.on_stage_end(Stage.TEST, avg_test_loss, None)
        self.step = 0
    
    
        return avg_test_loss,out
    
    
    def infer(self,test_set, max_key=None,min_key=None,progressbar=None,test_loader_kwargs={}):
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


    

def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    We expect `prepare_mini_librispeech` to have been called before this,
    so that the `train.json` and `valid.json` manifest files are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline. Adds noise, reverb, and babble on-the-fly.
    # Of course for a real enhancement dataset, you'd want a fixed valid set.
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("clean_sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        clean_sig = sb.dataio.dataio.read_audio(wav)
        return clean_sig

    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    hparams["dataloader_options"]["shuffle"] = False
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["id", "clean_sig"],
        ).filtered_sorted(sort_key="length")
    return datasets


# Recipe begins!
if __name__ == "__main__":
    #Sourav Change
    device="cpu"

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        sb.utils.distributed.run_on_main(
            prepare_mini_librispeech,
            kwargs={
                "data_folder": hparams["data_folder"],
                "save_json_train": hparams["train_annotation"],
                "save_json_valid": hparams["valid_annotation"],
                "save_json_test": hparams["test_annotation"],
            },
        )
    sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])

    # Create dataset objects "train" and "valid"
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    se_brain = SEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    print(hparams["model"])

    #se_brain.fit(
    #    epoch_counter=se_brain.hparams.epoch_counter,
    #    train_set=datasets["train"],
    #    valid_set=datasets["valid"],
    #    train_loader_kwargs=hparams["dataloader_options"],
    #    valid_loader_kwargs=hparams["dataloader_options"],
    #)

    # Load best checkpoint (highest STOI) for evaluation
    #test_stats,out = se_brain.evaluate(
    #    test_set=datasets["test"],
    #    max_key="stoi",
    #    test_loader_kwargs=hparams["dataloader_options"],
    #)
    #print("test_stats")
    #print(test_stats)
    #print("out--->")
    #print(out)
    
    #methods = inspect.getmembers(se_brain, predicate=inspect.ismethod)
    #for name, method in methods:
    #    print(f"Method: {name}")
    #    print(inspect.getsource(method))
    #    print('\n')
    
    
    #infer_stats=se_brain.infer(hparams["model"],datasets["valid"])
    #print(infer_stats)
    
    #speechbrain.inference.enhancement
    test_stats,out = se_brain.infer2(
        test_set=datasets["test"],
        max_key="stoi",
        test_loader_kwargs=hparams["dataloader_options"],
    )
