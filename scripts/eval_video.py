"""Stream annotated evaluation frames to one MP4 per episode."""
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from excavator_common.bridge_protocol import decode_camera_images, decode_rgb_payload


class EpisodeVideo:
    size = (1024, 864)

    def __init__(self, path, episode, seed, fps, max_steps, min_particles):
        self.path = Path(path)
        self.episode = episode
        self.seed = seed
        self.fps = float(fps)
        self.max_steps = max_steps
        self.min_particles = min_particles
        self.frame_count = 0
        self.writer = None
        self.overview = None
        self.cameras = {}
        self.stable_steps = 0
        try:
            self.font = ImageFont.truetype('DejaVuSans.ttf', 20)
        except OSError:
            self.font = ImageFont.load_default()

    def render(self, status, final=False):
        canvas = Image.new('RGB', self.size, (18, 22, 30))
        if self.overview is not None:
            canvas.paste(Image.fromarray(self.overview).resize((768, 768)), (0, 0))
        for index, name in enumerate(('0', '1', '2')):
            if name in self.cameras:
                image = Image.fromarray(self.cameras[name]).resize((256, 256))
                canvas.paste(image, (768, index * 256))
        draw = ImageDraw.Draw(canvas)
        for label, x, y, width in [('Overview', 0, 0, 170)] + [
            (f'Policy camera {index}', 768, index * 256, 256) for index in range(3)
        ]:
            draw.rectangle((x, y, x + width, y + 28), fill=(18, 22, 30))
            draw.text((x + 8, y + 2), label, font=self.font, fill='white')
        steps = int(status.get('policy_steps_executed', 0))
        count = int(status.get('final_newly_deposited_particle_count', 0))
        bucket = 'OPEN' if status.get('final_bucket_open') else 'CLOSED'
        draw.text((12, 774), f'Episode {self.episode}  |  Seed {self.seed}  |  '
                  f'Step {steps}/{self.max_steps}  |  Policy time {steps / self.fps:.1f}s',
                  font=self.font, fill='white')
        draw.text((12, 802), f'Deposited {count}/{self.min_particles} particles  |  '
                  f'Stable {self.stable_steps}/3  |  Bucket {bucket}', font=self.font, fill='white')
        if not final:
            result, color = 'RUNNING', (100, 190, 255)
        elif status.get('error') or status.get('status') == 'error':
            result, color = f"ERROR: {status.get('error') or 'Evaluation failed'}", (255, 130, 100)
        elif status.get('success'):
            complete = 'YES' if status.get('task_completed_success_and_bucket_open') else 'NO'
            result, color = f'SUCCESS  |  Task complete: {complete}', (90, 225, 130)
        else:
            result, color = f"FAILURE  |  {status.get('termination_reason', 'step_limit')}", (255, 190, 90)
        # Preserve the full error in the status JSON; fit its opening on screen.
        while len(result) > 3 and draw.textbbox((0, 0), result, font=self.font)[2] > 910:
            result = result[:-4] + '...'
        draw.text((12, 830), f'Result: {result}', font=self.font, fill=color)
        return np.asarray(canvas)

    def append(self, frame):
        if self.writer is None:
            self.writer = imageio_ffmpeg.write_frames(
                str(self.path), self.size, fps=self.fps, codec='libx264',
                pix_fmt_out='yuv420p', quality=7,
                output_params=['-movflags', '+faststart', '-threads', '2'],
            )
            self.writer.send(None)
        self.writer.send(np.ascontiguousarray(frame))
        self.frame_count += 1

    def write(self, reply, status, stable_steps=0):
        payload = reply.get('evaluation_video_frame')
        if payload is None:
            raise ValueError('Simulator did not return the evaluation overview frame')
        self.overview = decode_rgb_payload(payload, np_module=np)
        self.cameras = decode_camera_images(reply, np_module=np)
        self.stable_steps = stable_steps
        self.append(self.render(status))

    def finish(self, status):
        try:
            # Also produces a result video if startup failed before any image.
            frame = self.render(status, final=True)
            for _ in range(max(1, round(self.fps * 2))):
                self.append(frame)
        finally:
            if self.writer is not None:
                self.writer.close()
