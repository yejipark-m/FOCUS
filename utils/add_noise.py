import numpy as np
from PIL import Image
import skimage.util as sk


class NoiseProcessor:
    def __init__(self, mode='gaussian', scale=1.0, severity=1):
        self.mode = mode
        self.scale = scale
        self.severity = severity

    def _normalize(self, img: Image.Image):
        return np.array(img).astype(np.float32) / 255.0

    def _denormalize(self, arr: np.ndarray):
        return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        mode = self.mode
        scale = self.scale
        severity = self.severity

        if mode == 'black':
            return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))

        if mode == 'white':
            return Image.fromarray(np.full((h, w, 3), 255, dtype=np.uint8))

        if mode == 'gray':
            return Image.fromarray(np.full((h, w, 3), 127, dtype=np.uint8))

        if mode == 'random':
            return Image.fromarray(np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8))

        if mode == 'gaussian':
            noisy = np.random.normal(0, 1, size=(h, w, 3)).astype(np.float32)
            noisy = np.clip(noisy, 0, 1)
            return self._denormalize(noisy)

        if mode == 'uniform':
            noisy = np.random.uniform(0, 1, size=(h, w, 3)).astype(np.float32)
            noisy = np.clip(noisy, 0, 1)
            return self._denormalize(noisy)

        x = self._normalize(img)

        if mode == 'imagenet+gaussian':
            mean = np.array([0.485, 0.456, 0.406])
            stds = np.array([0.229, 0.224, 0.225])
            norm = (x - mean) / stds
            noise = np.random.normal(0, 1, size=norm.shape)
            out = np.clip(((1 - scale) * norm + scale * noise) * stds + mean, 0, 1)
            return self._denormalize(out)

        if mode == 'imagenet+uniform':
            mean = np.array([0.485, 0.456, 0.406])
            stds = np.array([0.229, 0.224, 0.225])
            norm = (x - mean) / stds
            noise = np.random.uniform(0, 1, size=norm.shape)
            out = np.clip(((1 - scale) * norm + scale * noise) * stds + mean, 0, 1)
            return self._denormalize(out)

        if mode == 'imagenet+diffusion':
            mean = np.array([0.485, 0.456, 0.406])
            stds = np.array([0.229, 0.224, 0.225])
            norm = (x - mean) / stds

            num_steps = 1000
            betas = 1e-5 + (0.5e-2 - 1e-5) / (1 + np.exp(-np.linspace(-6, 6, num_steps)))
            alphas = 1.0 - betas
            alpha_bars = np.cumprod(alphas)
            t = int(scale * (num_steps - 1))
            sqrt_alpha_bar = np.sqrt(alpha_bars[t])
            sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bars[t])
            noise = np.random.randn(*x.shape).astype(np.float32)
            out = np.clip(sqrt_alpha_bar * norm + sqrt_one_minus_alpha_bar * noise, 0, 1)
            return self._denormalize(out)

        if mode == 'shot':
            c = [60, 25, 12, 5, 3][severity - 1]
            mean = np.array([0.485, 0.456, 0.406])
            stds = np.array([0.229, 0.224, 0.225])
            norm = np.clip((x - mean) / stds, 0, 1)
            out = np.clip(np.random.poisson(norm * c) / c, 0, 1)
            return self._denormalize(out)

        if mode == 'impulse':
            c = [.03, .06, .09, 0.17, 0.27][severity - 1]
            mean = np.array([0.485, 0.456, 0.406])
            stds = np.array([0.229, 0.224, 0.225])
            norm = (x - mean) / stds
            out = sk.random_noise(norm, mode='s&p', amount=c)
            return self._denormalize(np.clip(out, 0, 1))

        if mode == 'speckle':
            c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
            mean = np.array([0.485, 0.456, 0.406])
            stds = np.array([0.229, 0.224, 0.225])
            norm = (x - mean) / stds
            out = np.clip(norm + norm * np.random.normal(scale=c, size=x.shape), 0, 1)
            return self._denormalize(out)

        raise ValueError(f"Unknown noise mode: {mode}")
