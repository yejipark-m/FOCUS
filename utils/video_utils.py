import io
from PIL import Image


def sample_frames(video_data, num_frames: int) -> list:
    """Sample `num_frames` PIL Images uniformly from a video.

    Accepts bytes, a file path (str), or an already-decoded list of PIL Images.
    Returns a list of RGB PIL Images of length <= num_frames.
    """
    if isinstance(video_data, list):
        frames = [
            f.convert("RGB") if hasattr(f, "convert") else Image.fromarray(f).convert("RGB")
            for f in video_data
        ]
        if len(frames) <= num_frames:
            return frames
        indices = [int(i * len(frames) / num_frames) for i in range(num_frames)]
        return [frames[i] for i in indices]

    import decord
    decord.bridge.set_bridge("native")
    buf = io.BytesIO(video_data) if isinstance(video_data, bytes) else video_data
    vr = decord.VideoReader(buf, ctx=decord.cpu(0))
    total = len(vr)
    indices = [min(int(i * total / num_frames), total - 1) for i in range(num_frames)]
    raw = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(f).convert("RGB") for f in raw]
