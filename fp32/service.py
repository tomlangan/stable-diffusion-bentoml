from contextlib import ExitStack
from starlette.middleware.cors import CORSMiddleware

import logging
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers.models import AutoencoderKL

from pydantic import BaseModel
import bentoml
from bentoml.io import Image, JSON, Multipart

class StableDiffusionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        model_id = "./models/Realistic_Vision_V2.0"
        vae_id = './models/sd-vae-ft-ema'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        vae = AutoencoderKL.from_pretrained(vae_id)
        txt2img_pipe = StableDiffusionPipeline.from_pretrained(model_id, vae=vae)

        self.txt2img_pipe = txt2img_pipe.to(self.device)

        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=txt2img_pipe.feature_extractor,
        ).to(self.device)

        self.inpaint_pipe = StableDiffusionInpaintPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to(self.device)

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def txt2img(self, data):
        prompt = data["prompt"]
        guidance_scale = data.get('guidance_scale', 7.5)
        height = data.get('height', 512)
        width = data.get('width', 512)
        num_inference_steps = data.get('num_inference_steps', 50)
        generator = torch.Generator(self.device)
        generator.manual_seed(data.get('seed'))

        if not data['safety_check']:
            self.txt2img_pipe.safety_checker = lambda images, **kwargs: (images, False)

        with ExitStack() as stack:
            if self.device != "cpu":
                _ = stack.enter_context(autocast(self.device))

            images = self.txt2img_pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images
            image = images[0]
            return image

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def img2img(self, init_image, data):
        logging.info(f"DATA: {data}")
        new_size = None
        longer_side = max(*init_image.size)
        if longer_side > 512:
            new_size = (512, 512)
        elif init_image.width != init_image.height:
            new_size = (longer_side, longer_side)

        if new_size:
            init_image =init_image.resize(new_size)

        prompt = data["prompt"]
        strength = data.get('strength', 0.8)
        guidance_scale = data.get('guidance_scale', 7.5)
        num_inference_steps = data.get('num_inference_steps', 50)
        generator = torch.Generator(self.device)
        generator.manual_seed(data.get('seed'))

        logging.info(f"DATA: {data}")
        logging.info(f"prompt: {prompt}")
        logging.info(f"strength: {strength}")
        logging.info(f"guidance_scale: {guidance_scale}")
        logging.info(f"num_inference_steps: {num_inference_steps}")
        logging.info(f"seed: {data['seed']}")

        if not data['safety_check']:
            self.img2img_pipe.safety_checker = lambda images, **kwargs: (images, False)

        with ExitStack() as stack:
            if self.device != "cpu":
                _ = stack.enter_context(autocast(self.device))

            images = self.img2img_pipe(
                prompt=prompt,
                init_image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images
            image = images[0]
            return image

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def inpaint(self, image, mask, data):
        prompt = data["prompt"]
        strength = data.get('strength', 0.8)
        guidance_scale = data.get('guidance_scale', 7.5)
        num_inference_steps = data.get('num_inference_steps', 50)
        generator = torch.Generator(self.device)
        generator.manual_seed(data.get('seed'))

        if not data['safety_check']:
            self.inpaint_pipe.safety_checker = lambda images, **kwargs: (images, False)

        with ExitStack() as stack:
            if self.device != "cpu":
                _ = stack.enter_context(autocast(self.device))

            images = self.inpaint_pipe(
                prompt=prompt,
                init_image=image,
                mask_image=mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images
            image = images[0]
            return image


stable_diffusion_runner = bentoml.Runner(StableDiffusionRunnable, name='stable_diffusion_runner', max_batch_size=10)

svc = bentoml.Service("stable_diffusion_fp32", runners=[stable_diffusion_runner])
svc.add_asgi_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["*"])


def generate_seed_if_needed(seed):
    if seed is None:
        generator = torch.Generator()
        seed = torch.seed()
    return seed

class Txt2ImgInput(BaseModel):
    prompt: str
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50
    safety_check: bool = True
    seed: int = None

@svc.api(input=JSON(pydantic_model=Txt2ImgInput), output=Image())
def txt2img(data, context):
    data = data.dict()
    data['seed'] = generate_seed_if_needed(data['seed'])
    image = stable_diffusion_runner.txt2img.run(data)
    for i in data:
        context.response.headers.append(i, str(data[i]))
    return image

class Img2ImgInput(BaseModel):
    prompt: str
    strength: float = 0.4
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    safety_check: bool = True
    seed: int = None

img2img_input_spec = Multipart(img=Image(), data=JSON(pydantic_model=Img2ImgInput))
@svc.api(input=img2img_input_spec, output=Image())
def img2img(img, data, context):
    logging.info(f"DATA1: {data}")
    data = data.dict()
    data['seed'] = generate_seed_if_needed(data['seed'])
    image = stable_diffusion_runner.img2img.run(img, data)
    for i in data:
        context.response.headers.append(i, str(data[i]))
    return image

inpaint_input_spec = Multipart(img=Image(), mask=Image(), data=JSON(pydantic_model=Img2ImgInput))
@svc.api(input=inpaint_input_spec, output=Image())
def inpaint(img, mask, data, context):
    data = data.dict()
    data['seed'] = generate_seed_if_needed(data['seed'])
    image = stable_diffusion_runner.inpaint.run(img, mask, data)
    for i in data:
        context.response.headers.append(i, str(data[i]))
    return image
