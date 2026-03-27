"""Fun-ASR-Nano 兼容注册模块。

用途：官方模型仓库当前未包含 `model.py`，而已发布的 funasr 包中也未稳定内置
`FunASRNano` 注册项。这里内置一份官方实现的兼容版，确保本地离线目录可加载。
"""

import logging
import os
import random
import re
import string
import time
import traceback
from itertools import groupby
from typing import Union

import torch
import torch.nn as nn
import torchaudio.functional as audio_functional
from funasr.metrics.compute_acc import compute_accuracy
from funasr.models.ctc.ctc import CTC
from funasr.register import tables
from funasr.train_utils.device_funcs import force_gatherable, to_device
from funasr.utils.datadir_writer import DatadirWriter
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video
from transformers import AutoConfig, AutoModelForCausalLM

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def forced_align(log_probs: torch.Tensor, targets: torch.Tensor, blank: int = 0):
    """兼容官方 tools.utils.forced_align。"""
    items = []
    try:
        log_probs, targets = log_probs.unsqueeze(0).cpu(), targets.unsqueeze(0).cpu()
        assert log_probs.shape[1] >= targets.shape[1]
        alignments, scores = audio_functional.forced_align(log_probs, targets, blank=blank)
        alignments, scores = alignments[0], torch.exp(scores[0]).tolist()
        for token, group in groupby(enumerate(alignments), key=lambda item: item[1]):
            if token == blank:
                continue
            grouped = list(group)
            start = grouped[0][0]
            end = start + len(grouped)
            score = max(scores[start:end])
            items.append({
                "token": token.item(),
                "start_time": start,
                "end_time": end,
                "score": round(score, 3),
            })
    except Exception:
        pass
    return items


@tables.register("model_classes", "FunASRNano")
class FunASRNano(nn.Module):
    """官方 Fun-ASR-Nano 的内置兼容实现。"""

    def __init__(
        self,
        audio_encoder: str = None,
        audio_encoder_conf: dict = None,
        audio_adaptor: str = None,
        audio_adaptor_conf: dict = None,
        llm: str = None,
        llm_conf: dict = None,
        input_size: int = 80,
        length_normalized_loss: bool = False,
        **kwargs,
    ):
        super().__init__()

        hub = (audio_encoder_conf or {}).get("hub", None)
        self.audio_encoder_activation_checkpoint = (audio_encoder_conf or {}).get(
            "activation_checkpoint", False
        )
        if hub == "ms":
            from funasr import AutoModel

            model = AutoModel(model=audio_encoder, model_revision="master")
            audio_encoder_output_size = (
                model.model.encoder_output_size if hasattr(model.model, "encoder_output_size") else -1
            )
            audio_encoder = model.model.model.encoder if hasattr(model.model, "model") else model.model.encoder
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **(audio_encoder_conf or {}))
            audio_encoder_output_size = audio_encoder.output_size()

        if (audio_encoder_conf or {}).get("freeze", True):
            for _, param in audio_encoder.named_parameters():
                param.requires_grad = False
            audio_encoder.eval()
        self.audio_encoder = audio_encoder

        init_param_path = (llm_conf or {}).get("init_param_path", None)
        llm_load_kwargs = (llm_conf or {}).get("load_kwargs", {})
        config = AutoConfig.from_pretrained(init_param_path)
        model = AutoModelForCausalLM.from_config(config, **llm_load_kwargs)
        if (llm_conf or {}).get("freeze", True):
            for _, param in model.named_parameters():
                param.requires_grad = False
            model.eval()
        if (llm_conf or {}).get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()
        self.llm_dtype = (llm_conf or {}).get("llm_dtype", "fp32")
        self.llm = model.to(dtype_map[self.llm_dtype])
        llm_dim = model.get_input_embeddings().weight.shape[-1]

        adaptor_class = tables.adaptor_classes.get(audio_adaptor)
        adaptor_conf = dict(audio_adaptor_conf or {})
        if audio_encoder_output_size > 0:
            adaptor_conf["encoder_dim"] = audio_encoder_output_size
        adaptor_conf["llm_dim"] = llm_dim if llm_dim is not None else adaptor_conf.get("llm_dim")
        audio_adaptor_model = adaptor_class(**adaptor_conf)
        if adaptor_conf.get("freeze", False):
            for _, param in audio_adaptor_model.named_parameters():
                param.requires_grad = False
            audio_adaptor_model.eval()
        self.audio_adaptor = audio_adaptor_model
        self.use_low_frame_rate = adaptor_conf.get("use_low_frame_rate", False)

        self.ctc_decoder = None
        self.ctc_tokenizer = None
        self.ctc = None
        self.blank_id = None
        ctc_decoder_class = tables.adaptor_classes.get(kwargs.get("ctc_decoder", None))
        if ctc_decoder_class is not None:
            try:
                ctc_tokenizer = kwargs.get("ctc_tokenizer", None) if "ctc_tokenizer" in kwargs else kwargs["dataset_conf"]["ctc_tokenizer"]
                ctc_tokenizer_conf = kwargs.get("ctc_tokenizer_conf", None) if "ctc_tokenizer_conf" in kwargs else kwargs["dataset_conf"]["ctc_tokenizer_conf"]
                if ctc_tokenizer is not None and ctc_tokenizer_conf is not None:
                    ctc_tokenizer_class = tables.tokenizer_classes.get(ctc_tokenizer)
                    self.ctc_tokenizer = ctc_tokenizer_class(**ctc_tokenizer_conf)
                assert ctc_tokenizer is not None, "ctc_tokenizer must be set"

                ctc_decoder_conf = kwargs.get("ctc_decoder_conf", {})
                if audio_encoder_output_size > 0:
                    ctc_decoder_conf["encoder_dim"] = audio_encoder_output_size
                self.ctc_decoder = ctc_decoder_class(**ctc_decoder_conf)
                init_param_path = ctc_decoder_conf.get("init_param_path", None)
                if init_param_path is not None:
                    src_state = torch.load(init_param_path, map_location="cpu")
                    flag = self.ctc_decoder.load_state_dict(src_state, strict=False)
                    logging.info(f"Loading ctc_decoder ckpt: {init_param_path}, status: {flag}")

                if ctc_decoder_conf.get("freeze", False):
                    for _, param in self.ctc_decoder.named_parameters():
                        param.requires_grad = False
                    self.ctc_decoder.eval()

                ctc_conf = kwargs.get("ctc_conf", {})
                ctc_vocab_size = kwargs.get("ctc_vocab_size", 60515)
                self.blank_id = ctc_conf.get("blank_id", ctc_vocab_size - 1)
                self.ctc_weight = kwargs.get("ctc_weight", 0.3)
                self.ctc = CTC(
                    odim=ctc_vocab_size,
                    encoder_output_size=audio_encoder_output_size,
                    blank_id=self.blank_id,
                    **ctc_conf,
                )
                self.detach_ctc_decoder = kwargs.get("detach_ctc_decoder", True)
                self.error_calculator = None
            except Exception as exc:
                logging.warning("FunASRNano compat 初始化 CTC 解码器失败，已降级为无 CTC 模式: %s", exc)
                self.ctc_decoder = None
                self.ctc_tokenizer = None
                self.ctc = None
                self.blank_id = None

        self.length_normalized_loss = length_normalized_loss
        rank = int(os.environ.get("RANK", 0))
        logging.info(f"rank: {rank}, FunASRNano compat model built")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("FunASRNano compat 仅用于推理")

    def encode(self, speech, speech_lengths):
        return self.audio_encoder(speech, speech_lengths)

    def data_template(self, data):
        system, user, assistant = [], [], []
        for item in data:
            role = item["role"]
            content = item["content"]
            if role == "system":
                system.append(content)
            elif role == "user":
                if "audio" in item:
                    content = [content, item["audio"]]
                user.append(content)
            elif role == "assistant":
                assistant.append(content)
        return {"system": system * len(user), "user": user, "assistant": assistant}

    def data_load_speech(self, contents: dict, tokenizer, frontend, meta_data=None, **kwargs):
        if meta_data is None:
            meta_data = {}
        pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        do_think = kwargs.get("dataset_conf", {}).get("do_think", True)
        sys_prompt = kwargs.get("dataset_conf", {}).get("sys_prompt", True)

        input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = [], [], [], [], [], [], []
        input_source_ids = []
        target_ids = []

        for i, (system_prompt, user_prompt, target_out) in enumerate(
            zip(contents["system"], contents["user"], contents["assistant"])
        ):
            if i >= kwargs.get("multiturn_num_max", 5) or len(input_ids) > kwargs.get("max_token_length", 1500):
                break
            if isinstance(user_prompt, (list, tuple)):
                user_prompt, audio = user_prompt
            else:
                audio = None

            if i == 0:
                source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                if not sys_prompt:
                    source_input = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                source_input = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            if not do_think:
                source_input += "<think>\n\n</think>\n\n"
            if kwargs.get("prev_text"):
                source_input += kwargs["prev_text"]

            splits = pattern.split(source_input)
            source_ids, fbank_mask_i = [], []
            fake_token_len_i, fbank_beg_i = 0, -1
            speech, speech_lengths = [], []

            for sub_str in splits:
                if not sub_str.startswith("<|startofspeech|>"):
                    sub_token = tokenizer.encode(sub_str)
                    source_ids += sub_token
                    fbank_mask_i += [0] * len(sub_token)
                    continue

                sub_str = sub_str.replace("<|startofspeech|>", "").replace("<|endofspeech|>", "")
                if not sub_str.startswith("!"):
                    continue
                sub_str = sub_str[1:]
                if sub_str.startswith("!"):
                    sub_str = audio

                time1 = time.perf_counter()
                data_src = load_audio_text_image_video(sub_str, fs=frontend.fs, **kwargs)
                time2 = time.perf_counter()
                meta_data["load_data"] = f"{time2 - time1:0.3f}"
                speech, speech_lengths = extract_fbank(
                    data_src,
                    data_type=kwargs.get("data_type", "sound"),
                    frontend=frontend,
                    is_final=True,
                )
                time3 = time.perf_counter()
                meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
                meta_data["batch_data_time"] = speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000

                if self.use_low_frame_rate:
                    olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                    olens = 1 + (olens - 3 + 2 * 1) // 2
                    fake_token_len_i = (olens - 1) // 2 + 1
                else:
                    fake_token_len_i = speech_lengths[0].item()
                fake_token = [0] * fake_token_len_i
                fbank_beg_i = len(source_ids)
                source_ids += fake_token
                fbank_mask_i += [1] * len(fake_token)

            fbank_beg.append(fbank_beg_i + len(input_ids))
            fake_token_len.append(fake_token_len_i)
            source_mask = [-100] * len(source_ids)
            target_out = f"{target_out}<|im_end|>"
            target_ids = tokenizer.encode(target_out)
            input_source_ids = input_ids + source_ids
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids
            fbank_mask += fbank_mask_i
            if len(speech) > 0:
                fbank.append(speech[0, :, :])
                fbank_lens.append(speech_lengths)

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64)
        attention_mask = torch.tensor([1] * len(input_ids_tensor), dtype=torch.int32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        fbank_mask_tensor = torch.tensor(fbank_mask, dtype=torch.float32)
        fbank_beg_tensor = torch.tensor(fbank_beg, dtype=torch.int32)
        fake_token_len_tensor = torch.tensor(fake_token_len, dtype=torch.int32)
        source_ids_tensor = torch.tensor(input_source_ids, dtype=torch.int64)
        target_ids_tensor = torch.tensor(target_ids, dtype=torch.int64)

        if len(fbank) > 0:
            speech_tensor = torch.nn.utils.rnn.pad_sequence(fbank, batch_first=True, padding_value=0.0)
            speech_lengths_tensor = torch.nn.utils.rnn.pad_sequence(fbank_lens, batch_first=True, padding_value=-1)
        else:
            speech_tensor = []
            speech_lengths_tensor = []

        return {
            "speech": speech_tensor,
            "speech_lengths": speech_lengths_tensor,
            "fbank_mask": fbank_mask_tensor[None, :],
            "fbank_beg": fbank_beg_tensor[None, :],
            "fake_token_len": fake_token_len_tensor[None, :],
            "input_ids": input_ids_tensor[None, :],
            "attention_mask": attention_mask[None, :],
            "labels_ids": labels_tensor,
            "source_ids": source_ids_tensor[None, :],
            "target_ids": target_ids_tensor[None, :],
        }

    def inference_prepare(self, data_in, data_lengths=None, key=None, tokenizer=None, frontend=None, **kwargs):
        meta_data = {}
        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        contents = self.data_template(data_in[0])
        output = self.data_load_speech(contents, tokenizer, frontend, meta_data=meta_data, **kwargs)
        batch = to_device(output, kwargs["device"])
        speech = batch["speech"]

        if len(speech) > 0:
            if "audio_embedding" in kwargs and "audio_embedding_lens" in kwargs:
                adaptor_out = kwargs["audio_embedding"]
                adaptor_out_lens = kwargs["audio_embedding_lens"]
                meta_data["encoder_out"] = adaptor_out
                meta_data["encoder_out_lens"] = adaptor_out_lens
            else:
                speech_lengths = batch["speech_lengths"][:, 0]
                if kwargs.get("fp16", False):
                    speech = speech.to(torch.float16)
                elif kwargs.get("bf16", False):
                    speech = speech.to(torch.bfloat16)
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
                adaptor_out, adaptor_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)
                meta_data["encoder_out"] = encoder_out
                meta_data["encoder_out_lens"] = encoder_out_lens
                meta_data["audio_adaptor_out"] = adaptor_out
                meta_data["audio_adaptor_out_lens"] = adaptor_out_lens
        else:
            adaptor_out = None
            adaptor_out_lens = None

        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        fbank_beg = batch["fbank_beg"]
        fake_token_len = batch["fake_token_len"]
        if not kwargs.get("teacherforcing", False):
            input_ids = source_ids

        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)

        if adaptor_out is not None:
            fake_token_len[fake_token_len < 0] = 0
            fbank_beg[fbank_beg < 0] = 0
            speech_idx = 0
            for batch_idx in range(inputs_embeds.shape[0]):
                for turn_id in range(fbank_beg.shape[1]):
                    fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                    if fbank_beg_idx <= 0:
                        continue
                    speech_token_len = fake_token_len[batch_idx, turn_id]
                    speech_token = adaptor_out[speech_idx, :speech_token_len, :]
                    try:
                        inputs_embeds[batch_idx, fbank_beg_idx:fbank_beg_idx + speech_token_len, :] = speech_token
                    except Exception:
                        logging.error(traceback.format_exc())
                        speech_token_len = adaptor_out_lens[speech_idx].item()
                        speech_token = adaptor_out[speech_idx, :speech_token_len, :]
                        inputs_embeds[batch_idx, fbank_beg_idx:fbank_beg_idx + speech_token_len, :] = speech_token
                    speech_idx += 1
        return inputs_embeds, contents, batch, source_ids, meta_data

    def get_prompt(self, hotwords: list[str], language: str = None, itn: bool = True):
        prompt = ""
        if len(hotwords) > 0:
            prompt = "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
            prompt += f"热词列表：[{', '.join(hotwords)}]\n"
        prompt += "语音转写" if language is None else f"语音转写成{language}"
        if not itn:
            prompt += "，不进行文本规整"
        return prompt + "："

    def generate_chatml(self, prompt: str, data: Union[str, torch.Tensor]):
        if isinstance(data, str):
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}<|startofspeech|>!{data}<|endofspeech|>"},
                {"role": "assistant", "content": "null"},
            ]
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}<|startofspeech|>!!<|endofspeech|>", "audio": data},
            {"role": "assistant", "content": "null"},
        ]

    def inference(self, data_in, data_lengths=None, key=None, tokenizer=None, frontend=None, **kwargs):
        prompt = self.get_prompt(kwargs.get("hotwords", []), kwargs.get("language", None), kwargs.get("itn", True))
        payload = [self.generate_chatml(prompt, data) for data in data_in]
        if key is None:
            chars = string.ascii_letters + string.digits
            key = ["rand_key_" + "".join(random.choice(chars) for _ in range(13)) for _ in payload]
        return self.inference_llm(payload, data_lengths=data_lengths, key=key, tokenizer=tokenizer, frontend=frontend, **kwargs)

    def inference_llm(self, data_in, data_lengths=None, key=None, tokenizer=None, frontend=None, **kwargs):
        inputs_embeds, contents, batch, source_ids, meta_data = self.inference_prepare(
            data_in, data_lengths, key, tokenizer, frontend, **kwargs
        )

        ctc_results = []
        if self.ctc_decoder is not None:
            encoder_out = meta_data["encoder_out"]
            encoder_out_lens = meta_data["encoder_out_lens"]
            decoder_out, decoder_out_lens = self.ctc_decoder(encoder_out, encoder_out_lens)
            ctc_logits = self.ctc.log_softmax(decoder_out)
            keys = key[0] if isinstance(key[0], (list, tuple)) else key
            if len(keys) < encoder_out.size(0):
                keys = keys * encoder_out.size(0)
            for i in range(encoder_out.size(0)):
                x = ctc_logits[i, : encoder_out_lens[i].item(), :]
                yseq = torch.unique_consecutive(x.argmax(dim=-1), dim=-1)
                mask = yseq != self.blank_id
                token_int = yseq[mask].tolist()
                text = self.ctc_tokenizer.decode(token_int)
                ctc_results.append({"key": keys[i], "text": text, "ctc_logits": x})

        llm_dtype = kwargs.get("llm_dtype", "fp32")
        if llm_dtype == "fp32":
            llm_dtype = "fp16" if kwargs.get("fp16", False) else llm_dtype
            llm_dtype = "bf16" if kwargs.get("bf16", False) else llm_dtype

        device_type = torch.device(kwargs.get("device", "cuda")).type
        with torch.autocast(
            device_type=device_type if device_type in ["cuda", "xpu", "mps"] else "cpu",
            enabled=llm_dtype != "fp32",
            dtype=dtype_map[llm_dtype],
        ):
            label = contents["assistant"][-1]
            self.llm = self.llm.to(dtype_map[llm_dtype])
            inputs_embeds = inputs_embeds.to(dtype_map[llm_dtype])
            llm_kwargs = kwargs.get("llm_kwargs", {})
            attention_mask = batch.get("attention_mask", None)
            generated_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=kwargs.get("max_length", 512),
                pad_token_id=self.llm.config.pad_token_id or self.llm.config.eos_token_id,
                **llm_kwargs,
            )
            response = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=kwargs.get("skip_special_tokens", True),
            )[0]

        response = kwargs.get("prev_text", "") + response
        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"{0 + 1}best_recog"]

        results = []
        response_clean = re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", response)
        result_i = {
            "key": key[0],
            "text": re.sub(r"\s+", " ", response.replace("/sil", " ")),
            "text_tn": response_clean,
            "label": label,
        }
        results.append(result_i)

        for ctc_result, result in zip(ctc_results, results):
            result["ctc_text"] = ctc_result["text"].replace("<|nospeech|>", "")
            target_ids = torch.tensor(self.ctc_tokenizer.encode(result["ctc_text"]), dtype=torch.int64)
            result["ctc_timestamps"] = forced_align(ctc_result["ctc_logits"], target_ids, self.blank_id)
            target_ids = torch.tensor(self.ctc_tokenizer.encode(result["text"]), dtype=torch.int64)
            result["timestamps"] = forced_align(ctc_result["ctc_logits"], target_ids, self.blank_id)
            for timestamps in [result["timestamps"], result["ctc_timestamps"]]:
                for timestamp in timestamps:
                    timestamp["token"] = self.ctc_tokenizer.decode([timestamp["token"]])
                    timestamp["start_time"] = timestamp["start_time"] * 6 * 10 / 1000
                    timestamp["end_time"] = timestamp["end_time"] * 6 * 10 / 1000

        if ibest_writer is not None:
            ibest_writer["text"][key[0]] = response.replace("\n", " ")
            ibest_writer["label"][key[0]] = label.replace("\n", " ")
            ibest_writer["text_tn"][key[0]] = response_clean
        return results, meta_data

    @staticmethod
    def from_pretrained(model: str = None, **kwargs):
        from funasr import AutoModel

        built_model, built_kwargs = AutoModel.build_model(model=model, **kwargs)
        return built_model, built_kwargs