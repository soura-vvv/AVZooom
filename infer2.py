
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
import numpy as np
import scipy.io.wavfile as wavf


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
        #print("Batchsdhosijd---------------------")
        #print(batch)
        batch = batch.to(self.device)
        
        
        #self.clean_wavs, self.lens = batch.clean_sig
        self.clean_wavs, self.lens = batch.clean_sig
        #print("Clean_WAVS--")
        #print(self.clean_wavs)
        #print("Lens--")
        #print(self.lens)
        
        noisy_wavs, self.lens = self.hparams.wav_augment(
            self.clean_wavs, self.lens
        )
        #Use this for augmenting noise
        #noisy_wavs=self.clean_wavs

        noisy_feats = self.compute_feats(noisy_wavs)

        #print("Noisy Feats Size")
        #print(noisy_feats.size())
        # Masking is done here with the "signal approximation (SA)" algorithm.
        # The masked input is compared directly with clean speech targets.
        
        #Appending Coordinates
        coordinates=torch.tensor(batch.coordinates).to(device)
        coordinates=coordinates.unsqueeze(1)
        coordinates=coordinates.repeat(1,noisy_feats.size(dim=1),1)
        
        noisy_feats_coordinates=torch.cat((noisy_feats,coordinates),2).to(device)
        
        
        #Actual Training
        mask = self.modules.model(noisy_feats_coordinates)
        
        
        #make mask 259
        #temp_zeros=torch.zeros(mask.size(dim=0),mask.size(dim=1),2).to(device)
        #mask=torch.cat((mask,temp_zeros),2).to(device)
        
        #noisy_feats_chopped=torch.split(noisy_feats,257,dim=2)
        #print("New Dimension:")
        #print(noisy_feats)
        #print("Mask Time:")
        #print(mask.size())
        
        predict_spec = torch.mul(mask, noisy_feats)
        
        #predict_spec=torch.mul(mask,noisy_feats)
        # Also return predicted wav, for evaluation. Note that this could
        # also be used for a time-domain loss term.
        
        #predict_spec_chopped=torch.split(predict_spec,257,dim=2)
        #print("Predict_SpecSize")
        #print(predict_spec.size())
        #print("Predict_Spec_ChoppedSize")
        #print(predict_spec_chopped[0].size())
        #print("Noisy Wavs Size")
        #print(noisy_wavs.size())
        #print("Noisy Wavs Size")
        #print(noisy_wavs.size())
        predict_wav = self.hparams.resynth(
            torch.expm1(predict_spec), noisy_wavs
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
        #temp_zeros=torch.zeros(feats.size(dim=0),feats.size(dim=1),2).to(device)
        #print("Temp_Zeros Size:")
        #print(temp_zeros.size())
        #print("Feats Size:")
        #print(feats.size())
        #feats=torch.cat((feats,temp_zeros),2).to(device)
        #print("Post Feats Size:")
        #print(feats.size())
        #Sourav Change End
        return feats

   
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
        #loss = self.compute_objectives(out, batch, stage=stage)
        return out
    # Define the inference function
    def infer2(self,test_set, max_key=None,min_key=None,progressbar=None,test_loader_kwargs={}):
        out=[]
        if progressbar is None:
            progressbar = not self.noprogressbar

        test_loader_kwargs["ckpt_prefix"] = None
        test_set = self.make_dataloader(
            test_set, sb.Stage.TEST, **test_loader_kwargs
        )
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
                outz = self.evaluate_batch(batch, stage="")
                out.append(outz)
                #avg_test_loss = self.update_average(loss, avg_test_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break
                #exit()
                
                

            #self.on_stage_end(Stage.TEST, avg_test_loss, None)
        self.step = 0
    
    
        return out
    
    
    
        


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
        "train":hparams["train_annotation"],
        "valid":hparams["valid_annotation"],
        "test":hparams["test_annotation"],
        "infer":hparams["infer_annotation"]
    }
    hparams["dataloader_options"]["shuffle"] = False
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["id", "clean_sig","coordinates"],
        ).filtered_sorted(sort_key="length")
    return datasets

def write_out_audio(audio_wavs):
    sample_rate=hparams["sample_rate"]
    #samples=audio_wavs[0].cpu().numpy()
    
    #wavf.write("outputs/NoisyTestInference0.wav",sample_rate,samples)
    #samples=audio_wavs[1].cpu().numpy()
    
    #wavf.write("outputs/NoisyTestInference1.wav",sample_rate,samples)
    #samples=audio_wavs[2].cpu().numpy()
    
    #wavf.write("outputs/NoisyTestInference2.wav",sample_rate,samples)
    i=0
    for audios in audio_wavs:
        fileName="outputs/"+"TestInference"+str(i)+".wav"
        wavf.write(fileName,sample_rate,audios)
        i+=1

# Recipe begins!
if __name__ == "__main__":
    #Sourav Change
    device="cuda"

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

    
    se_brain = SEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    for sets in datasets["infer"]:
        print(sets)
    print(hparams["model"])
    print(datasets["train"])
    out = se_brain.infer2(
        test_set=datasets["infer"],
        max_key="stoi",
        test_loader_kwargs=hparams["dataloader_options"],
    )
    print("=============================================================================")
    print(hparams["train_annotation"])
    #print(*out,sep="\n")
    #print(len(out[0]['wav'][0]))
    #wavzout=out[0]['wav'][0]
    #samples=wavzout.cpu().numpy()
    #write("inferenceOut1.wav", hparams["sample_rate"], samples.astype(np.int16))
    print(out[0]['wav'].size())
    write_out_audio(out[0]['wav'])
    
    #/home/sxp3410/Masters/speechbrain/templates/enhancement/AVZooom
