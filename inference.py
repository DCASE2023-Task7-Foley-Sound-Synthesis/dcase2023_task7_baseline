from typing import List
from numpy import ndarray
from torch import Tensor
from abc import ABC, abstractmethod

import os
import argparse
import math
import time
import datetime
import torch
from tqdm import tqdm
import soundfile as sf

from vqvae import VQVAE
from pixelsnail import PixelSNAIL
from HiFiGanWrapper import HiFiGanWrapper


class SoundSynthesisModel(ABC):
    @abstractmethod
    def synthesize_sound(self, class_id: str, number_of_sounds: int) -> List[ndarray]:
        raise NotImplementedError


class DCASE2023FoleySoundSynthesis:
    def __init__(
        self, number_of_synthesized_sound_per_class: int = 100, batch_size: int = 16
    ) -> None:
        self.number_of_synthesized_sound_per_class: int = (
            number_of_synthesized_sound_per_class
        )
        self.batch_size: int = batch_size
        self.class_id_dict: dict = {
            0: 'DogBark',
            1: 'Footstep',
            2: 'GunShot',
            3: 'Keyboard',
            4: 'MovingMotorVehicle',
            5: 'Rain',
            6: 'Sneeze_Cough',
        }
        self.sr: int = 22050
        self.save_dir: str = "./synthesized"

    def synthesize(self, synthesis_model: SoundSynthesisModel) -> None:
        for sound_class_id in self.class_id_dict:
            sample_number: int = 1
            save_category_dir: str = (
                f'{self.save_dir}/{self.class_id_dict[sound_class_id]}'
            )
            os.makedirs(save_category_dir, exist_ok=True)
            for _ in tqdm(
                range(
                    math.ceil(
                        self.number_of_synthesized_sound_per_class / self.batch_size
                    )
                ),
                desc=f"Synthesizing {self.class_id_dict[sound_class_id]}",
            ):
                synthesized_sound_list: list = synthesis_model.synthesize_sound(
                    sound_class_id, self.batch_size
                )
                for synthesized_sound in synthesized_sound_list:
                    if sample_number <= self.number_of_synthesized_sound_per_class:
                        sf.write(
                            f"{save_category_dir}/{str(sample_number).zfill(4)}.wav",
                            synthesized_sound,
                            samplerate=self.sr,
                        )
                        sample_number += 1


# ================================================================================================================================================
class BaseLineModel(SoundSynthesisModel):
    def __init__(
        self, pixel_snail_checkpoint: str, vqvae_snail_checkpoint: str
    ) -> None:
        super().__init__()
        self.pixel_snail = PixelSNAIL(
            [20, 86],
            512,
            256,
            5,
            4,
            4,
            256,
            dropout=0.1,
            n_cond_res_block=3,
            cond_res_channel=256,
        )
        self.pixel_snail.load_state_dict(
            torch.load(pixel_snail_checkpoint, map_location='cpu')['model']
        )
        self.pixel_snail.cuda()
        self.pixel_snail.eval()

        self.vqvae = VQVAE()
        self.vqvae.load_state_dict(
            torch.load(vqvae_snail_checkpoint, map_location='cpu')
        )
        self.vqvae.cuda()
        self.vqvae.eval()

        self.hifi_gan = HiFiGanWrapper(
            './checkpoint/hifigan/g_00935000',
            './checkpoint/hifigan/hifigan_config.json',
        )

    @torch.no_grad()
    def synthesize_sound(self, class_id: str, number_of_sounds: int) -> List[ndarray]:
        audio_list: List[ndarray] = list()

        feature_shape: list = [20, 86]
        vq_token: Tensor = torch.zeros(
            number_of_sounds, *feature_shape, dtype=torch.int64
        ).cuda()
        cache = dict()

        for i in tqdm(range(feature_shape[0]), desc="pixel_snail"):
            for j in range(feature_shape[1]):
                out, cache = self.pixel_snail(
                    vq_token[:, : i + 1, :],
                    label_condition=torch.full([number_of_sounds, 1], int(class_id))
                    .long()
                    .cuda(),
                    cache=cache,
                )
                prob: Tensor = torch.softmax(out[:, :, i, j], 1)
                vq_token[:, i, j] = torch.multinomial(prob, 1).squeeze(-1)

        pred_mel = self.vqvae.decode_code(vq_token).detach()
        for j, mel in enumerate(pred_mel):
            audio_list.append(self.hifi_gan.generate_audio_by_hifi_gan(mel))
        return audio_list


# ===============================================================================================================================================
if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vqvae_checkpoint', type=str, default='./checkpoint/vqvae/vqvae.pth'
    )
    parser.add_argument(
        '--pixelsnail_checkpoint',
        type=str,
        default='./checkpoint/pixelsnail-final/bottom_1400.pt',
    )
    parser.add_argument(
        '--number_of_synthesized_sound_per_class', type=int, default=100
    )
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    dcase_2023_foley_sound_synthesis = DCASE2023FoleySoundSynthesis(
        args.number_of_synthesized_sound_per_class, args.batch_size
    )
    dcase_2023_foley_sound_synthesis.synthesize(
        synthesis_model=BaseLineModel(args.pixelsnail_checkpoint, args.vqvae_checkpoint)
    )
    print(str(datetime.timedelta(seconds=time.time() - start)))
