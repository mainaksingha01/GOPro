import os.path as osp
import os
from collections import OrderedDict
import math
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import datetime
import time 
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torch.nn.modules.loss import _Loss
import augmix_ops as augmentations

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import MetricMeter, AverageMeter, load_pretrained_weights, load_checkpoint, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from utils import *

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
df = pd.DataFrame()
cls_label = pd.DataFrame()


device_cuda = "cuda"


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class AdaIN_trans(nn.Module):
		def __init__(self):
				super().__init__()
		def mu(self, x):
				return torch.sum(x,(1))/(x.shape[1])
		def sigma(self, x):
				return torch.sqrt((torch.sum((x.permute([1,0,2])-self.mu(x)).permute([1,0,2])**2,(1))+0.000000023)/(x.shape[1]))


class style_projector(nn.Module):
    def __init__(self):
        super(style_projector,self).__init__()
        self.linear = nn.ModuleList(nn.Linear(768,512) for _ in range (12))
        self.adain=AdaIN_trans()
        self.gap=nn.AdaptiveAvgPool2d((1,768))
    def forward(self,data):
        data_prompt=[]
        for i in range(len(data)):
            x_mu=self.adain.mu(data[i]).unsqueeze(1).to(torch.float32)
            x_sigma=self.adain.sigma(data[i]).unsqueeze(1).to(torch.float32)
            x_cat = torch.cat((x_mu, x_sigma),1)
            x_cat = self.gap(x_cat).squeeze(1)
            x_final = self.linear[i](x_cat)
            data_prompt.append(x_final)
        output = torch.stack(data_prompt, dim=1)
        return output


class content_projector(nn.Module):
    def __init__(self):
        super(content_projector,self).__init__()
        self.linear = nn.ModuleList(nn.Linear(768,512) for _ in range (12))
        self.adain=AdaIN_trans()
        self.gap=nn.AdaptiveAvgPool2d((1,768))
    def forward(self,data):
        data_prompt=[]
        for i in range(len(data)):
            x_gap = self.gap(data[i]).squeeze(1)
            x_lin=self.linear[i](x_gap)
            data_prompt.append(x_lin)
        output = torch.stack(data_prompt, dim=1)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x,_ = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.GOPRO.N_CTX
        ctx_init = cfg.TRAINER.GOPRO.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        self.style = style_projector()
        self.content = content_projector()
        self.frg = nn.AdaptiveAvgPool2d((1,vis_dim))
        self.rho = nn.Linear(12,1)

        
        if cfg.TRAINER.GOPRO.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) 
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :]) 
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts 
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  
                ctx,   
                suffix, 
            ],
            dim=1,
        )

        return prompts
    
    def forward(self, data):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  
        n_ctx = self.n_ctx        
    
        style_feat = self.style(data).unsqueeze(2)
        content_feat = self.content(data).unsqueeze(2)
        mixed_feat = torch.cat((style_feat, content_feat),2)
        stylecontent_feat = self.frg(mixed_feat).squeeze(2)
        stylecontent_feat = stylecontent_feat.permute(0,2,1)
        output = []
        for i in range(n_ctx):   
            x = self.rho(stylecontent_feat).permute(0,2,1)
            x = x.squeeze(1)
            output.append(x)
        feat_tokens = torch.stack(output, dim=1)
        
        ctx = ctx.unsqueeze(0)         
        ctx_shifted = ctx + feat_tokens       
        
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix) 
            prompts.append(pts_i)
        prompts = torch.stack(prompts)        
        return prompts
    
class sim_projector(nn.Module):
    def __init__(self):
        super(sim_projector,self).__init__()
        self.linear = nn.Linear(512,512)
        self.batchnorm = nn.BatchNorm1d(512)
    def forward(self,image):
        x = self.linear(image)
        output = self.batchnorm(x)
        return output
    

def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return views


class CustomCLIP(nn.Module): 
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.simproj = sim_projector()

    # MOCOv3 Augmentations
    def mocov3(self,images):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        images = images.to(device_cuda) 
        augmented_images = []
        for i in range(images.shape[0]):
            image = images[i]
            image = transforms.functional.to_pil_image(image.cpu())
            augmented_images.append(transform(image).to(device_cuda) )
        return torch.stack(augmented_images) 
    

    # Augmix transformations
    def augmixtrans(self,images):
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
        
        base_transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224)])
        preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        transform = AugMixAugmenter(base_transform, preprocess, n_views=images.size(0), augmix=True)

        images = images.to(device_cuda) 
        augmented_images = []
        for i in range(images.shape[0]):
            image = images[i]
            image = transforms.functional.to_pil_image(image.cpu())
            augmented_images.append(transform(image))
        for i in range(len(augmented_images)):
            aug_tensor = torch.stack(augmented_images[i]).to(device_cuda) 
            return aug_tensor
            
    
    
    @autocast()    
    def forward(self, image, label):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # original image
        image_features, data = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(data)
        logits = []
        image_textfeat = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_textfeat.append(text_features)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        image_textfeat = torch.stack(image_textfeat)
        logits = torch.stack(logits)

        # Mocov3
        mocov3_aug = self.mocov3(image)
        mocov3_features, mocov3_data = self.image_encoder(mocov3_aug.type(self.dtype))
        mocov3_features = mocov3_features / mocov3_features.norm(dim=-1, keepdim=True) 
        mocov3_prompts = self.prompt_learner(mocov3_data)
        mocov3_textfeat = []
        for pts_i in mocov3_prompts:
            text_feat = self.text_encoder(pts_i, tokenized_prompts)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            mocov3_textfeat.append(text_feat)
        mocov3_textfeat = torch.stack(mocov3_textfeat)


        # Augmix
        augmix_aug = self.augmixtrans(image)
        augmix_features, augmix_data = self.image_encoder(augmix_aug.type(self.dtype))
        augmix_features = augmix_features / augmix_features.norm(dim=-1, keepdim=True)
        augmix_prompts = self.prompt_learner(augmix_data)
        augmix_textfeat = []
        for pts_i in augmix_prompts:
            text_feat = self.text_encoder(pts_i, tokenized_prompts)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            augmix_textfeat.append(text_feat)
        augmix_textfeat = torch.stack(augmix_textfeat)

        image_simproj = self.simproj(image_features)
        mocov3_simproj = self.simproj(mocov3_features)

        return logits, image_textfeat, mocov3_textfeat, augmix_textfeat, image_simproj, mocov3_simproj, label


class SIMCLRLoss(nn.Module):

    def __init__(self, temperature=0.1): 
        super().__init__()
        self.tau = temperature
        self.labels = None
        self.masks = None
        self.last_local_batch_size = None

    def forward(self, image1, image2):
        image1 = F.normalize(image1, dim=-1, p=2)
        image2 = F.normalize(image2, dim=-1, p=2)

        local_batch_size = image1.size(0) 

        k_a, k_b = all_gather_batch_with_grad([image1, image2])

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(
                local_batch_size, device=image1.device
            )
            total_batch_size = local_batch_size * get_world_size()
            self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
            self.last_local_batch_size = local_batch_size

        logits_aa = torch.matmul(image1, k_a.transpose(0, 1)) / self.tau
        logits_aa = logits_aa - self.masks
        logits_bb = torch.matmul(image2, k_b.transpose(0, 1)) / self.tau
        logits_bb = logits_bb - self.masks
        logits_ab = torch.matmul(image1, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(image2, k_a.transpose(0, 1)) / self.tau

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), self.labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), self.labels)
        loss = (loss_a + loss_b) / 2

        return loss

class GOProLoss(_Loss):
    def __init__(self, T):
        super(GOProLoss, self).__init__() 
        self.T = T
        self.simloss = SIMCLRLoss()

    def forward(self, logits, image_textfeat, mocov3_textfeat, augmix_textfeat, image_simproj, mocov3_simproj, label):
        ce_loss = F.cross_entropy(logits, label)  
        sem_loss = F.mse_loss(image_textfeat, mocov3_textfeat) + F.mse_loss(image_textfeat, augmix_textfeat)
        con_loss = self.simloss(image_simproj, mocov3_simproj)
        total_loss = ce_loss + sem_loss + 0.1*con_loss
        return total_loss

@TRAINER_REGISTRY.register()
class GOPro(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.GOPRO.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.GOPRO.PREC == "fp32" or cfg.TRAINER.GOPRO.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = ["prompt_learner", "simproj"]
        
        for name, param in self.model.named_parameters():
            if not any(n in name for n in name_to_update):
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
       
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM) 
        self.register_model("gopro", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.GOPRO.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)          

        if cfg.LOSS.NAME == "gopro":
            self.criterion = GOProLoss(T=cfg.LOSS.T)
        else:
            raise NotImplementedError


    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def train(self):
        """Generic training loops."""

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()


    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.GOPRO.PREC
        if prec == "amp":
            with autocast():
                logits, image_textfeat, mocov3_textfeat, augmix_textfeat, image_simproj, mocov3_simproj, label = model(image, label)
                total_loss = self.criterion(logits, image_textfeat, mocov3_textfeat, augmix_textfeat, image_simproj, mocov3_simproj, label)
            optim.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits, image_textfeat, mocov3_textfeat, augmix_textfeat, image_simproj, mocov3_simproj, label = model(image, label)
            optim.zero_grad()
            total_loss = self.criterion(logits, image_textfeat, mocov3_textfeat, augmix_textfeat, image_simproj, mocov3_simproj, label)
            total_loss.sum().backward()
            optim.step()

        loss_summary = {"loss": total_loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr() 

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) %
                                self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if
                                self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if do_test:
            curr_result = self.test()
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(self.epoch,
                                self.output_dir,
                                model_name="model-best.pth.tar")

            self.set_model_mode("train")

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()

    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            logits, image_textfeat, mocov3_textfeat, augmix_textfeat, image_simproj, mocov3_simproj, label = self.model_inference(input, label)
            self.evaluator.process(logits, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
