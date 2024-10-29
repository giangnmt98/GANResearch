from abc import ABC

from ganresearch.models.base_gan import BaseGAN


class WGANGP(BaseGAN, ABC):
    def __init__(self, config):
        super().__init__(config)
