import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


class SEBrain(sb.Brain):

    def enhance_dataset(
        self,
        dataset, # Must be obtained from the dataio_function
        max_key, # We load the model with the max STOI
        loader_kwargs # opts for the dataloading
    ):

    # If dataset isn't a Dataloader, we create it.
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, Stage.TEST, **loader_kwargs
            )


        self.on_evaluate_start(max_key=max_key) # We call the on_evaluate_start that will load the best model
        self.modules.eval() # We set the model to eval mode (remove dropout etc)

    # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():

            transcripts = []
            for batch in tqdm(dataset, dynamic_ncols=True):

                # Make sure that your compute_forward returns the predictions !!!
                # In the case of the template, when stage = TEST, a beam search is applied
                # in compute_forward().
                out = self.compute_forward(batch, stage=sb.Stage.TEST)
                p_seq, wav_lens, predicted_tokens = out

                # We go from tokens to words.
                predicted_words = self.tokenizer(
                    predicted_tokens, task="decode_from_list"
                )
                transcripts.append(predicted_words)

        return transcripts



if __name__ == "__main__":
    #Sourav Change
    device="cuda"

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    
    predictions = se_brain.enhance_dataset(
        dataset=datasets["valid"], # Must be obtained from the dataio_function
        max_key="stoi", # We Load best checkpoint (highest STOI) for evaluation
        loader_kwargs=hparams["transcribe_dataloader_opts"], # opts for the dataloading
    )
    print("predictions")
    print(predictions)

