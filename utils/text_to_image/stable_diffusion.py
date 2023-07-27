

from evaluate import load
from datasets import load_dataset
from utils.misc import OutOfInstances
from utils.helpers import DatasetStub


from text_to_image.stable_diffusion.stablediffusion.ldm.models.diffusion.ddim import DDIMSampler


class StableDiffusion(DatasetStub):

    def __init__(self):
        self.available_instances = len(self._librispeech["audio"])
        self._idx = 0
        self._transcriptions = []

    def get_input_array(self):
        try:
            return self._librispeech["audio"][self._idx]["array"]
        except IndexError:
            raise OutOfInstances

    def submit_transcription(self, text: str):
        self._transcriptions.append(text)
        self._idx += 1

    def reset(self):
        self._idx = 0
        self._transcriptions = []
        return True

    def summarize_accuracy(self):
        assert len(self._transcriptions) == len(self._librispeech["text"][:self._idx])
        wer_score = load("wer").compute(
            references=self._librispeech["text"][:self._idx], predictions=self._transcriptions
        )
        print("\n  WER score = {:.3f}".format(wer_score))
        print(f"\n  Accuracy figures above calculated on the basis of {self._idx} sample(s).")
        return {"wer_score": wer_score}


    def preprocess(self):

        sampler = DDIMSampler(model, device=device)

        os.makedirs(opt.outdir, exist_ok=True)
        outpath = opt.outdir

        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "SDV2"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

        batch_size = opt.n_samples
        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
        if not opt.from_file:
            prompt = opt.prompt
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {opt.from_file}")
            with open(opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = [p for p in data for i in range(opt.repeat)]
                data = list(chunk(data, batch_size))

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        sample_count = 0
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

        start_code = None

        with torch.no_grad(), \
                precision_scope(opt.device), \
                model.ema_scope():
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples, _ = sampler.sample(S=opt.steps,
                                                conditioning=c,
                                                batch_size=opt.n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=uc,
                                                eta=opt.ddim_eta,
                                                x_T=start_code)

                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img = put_watermark(img, wm_encoder)
                        img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                        base_count += 1
                        sample_count += 1

                    all_samples.append(x_samples)

            # additionally, save as grid
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=n_rows)

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid = Image.fromarray(grid.astype(np.uint8))
            grid = put_watermark(grid, wm_encoder)
            grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
            grid_count += 1