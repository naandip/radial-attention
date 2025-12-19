import os
import time
import csv
import torch
import numpy as np

# --- Imports from the environment ---
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from transformers import UMT5EncoderModel

# Assumes radial_attn is in your path
from radial_attn.utils import set_seed
from radial_attn.models.wan.inference import replace_wan_attention
from radial_attn.models.wan.sparse_transformer import replace_sparse_forward

# --- Configuration ---
prompts = [
    # 1. Viscous Flow
    "A close-up, high-detail shot of thick, molten iridescent lava flowing slowly over a jagged obsidian rock. The liquid has a heavy, syrupy consistency, with glowing orange light peeking through cracks as the surface cools into a dark crust.",
    # 2. Granular Collapse
    "An hourglass filled with glowing neon blue sand. The sand is falling rapidly through the narrow neck, creating a turbulent, grainy splash at the bottom that builds up into a shimmering pile with realistic physics.",
    # 3. Aerodynamic Ripple
    "A slow-motion shot of a wide, crimson silk banner snapping and rippling in a heavy wind against a stormy grey sky. The fabric shows intricate weave textures and realistic folds as the wind waves travel through it.",
    # 4. Macro Biological
    "A time-lapse macro shot of a deep purple orchid blooming. The petals slowly unfurl and expand, revealing a velvety texture and tiny droplets of morning dew that roll down the surface as the flower opens.",
    # 5. Specular Metamorphosis
    "A sphere of liquid chrome floating in a white void. The sphere undulates and morphs into various geometric shapes, its highly reflective surface mirroring a distorted studio lighting setup.",
    # 6. High-Speed Impact
    "A crystal glass filled with splashing milk shattering on a dark marble floor. Shards of glass and white liquid fly outward in an explosive, radial motion captured in extreme slow motion.",
    # 7. Atmospheric Swirl
    "Thick, multicolored plumes of ink being dropped into clear water. The ink swirls in complex, chaotic patterns, expanding and thinning out into delicate, smoky threads that fill the frame.",
    # 8. Fur Rippling
    "A close-up of a snow leopard shaking its head to remove water. The thick, spotted fur ripples in waves starting from the neck to the ears, with fine water droplets being flung out in a circular mist.",
    # 9. Luminous Particle Dance
    "A swarm of golden fireflies dancing inside a dark forest at night. The fireflies move in erratic, organic paths, leaving behind faint trails of glowing light that illuminate the rough bark of nearby trees.",
    # 10. Elastic Bounce
    "A bright yellow rubber ducky falling onto a soft, squishy jelly surface. The jelly deforms deeply upon impact and wobbles with rhythmic, dampened vibrations while the ducky bounces back up."
]

# Config
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" 
device = "cuda"
height = 480
width = 640
flow_shift = 5.0
guidance_scale = 5.0
num_inference_steps = 50
seed = 0

csv_filename = "generation_metrics.csv"
output_dir = "comparision_videos"
os.makedirs(output_dir, exist_ok=True)

# Frame count calculation
num_prompts = len(prompts)
scales = np.linspace(0, 1, num_prompts)**3 
frame_counts = [59, 61, 71, 81, 91, 121, 151, 161, 191, 241]

# --- 1. Load Model Once ---
print(f"Loading model: {model_id}...")
set_seed(seed)
replace_sparse_forward() 

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)

scheduler = UniPCMultistepScheduler(
    prediction_type='flow_prediction', 
    use_flow_sigmas=True, 
    num_train_timesteps=1000, 
    flow_shift=flow_shift
)

pipe = WanPipeline.from_pretrained(
    model_id, 
    text_encoder=text_encoder, 
    transformer=transformer, 
    vae=vae, 
    torch_dtype=torch.bfloat16
)
pipe.scheduler = scheduler
pipe.to(device)
print("Model loaded successfully.")

# Default Negative Prompt
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

# --- 2. Define Workload ---
work_queue = []

# Pass 1: Add all DENSE jobs
for i, prompt in enumerate(prompts):
    work_queue.append({
        "prompt_idx": i,
        "prompt": prompt,
        "frames": int(frame_counts[i]),
        "pattern": "dense"
    })

# # Pass 2: Add all RADIAL jobs
# for i, prompt in enumerate(prompts):
#     work_queue.append({
#         "prompt_idx": i,
#         "prompt": prompt,
#         "frames": int(frame_counts[i]),
#         "pattern": "radial"
#     })

# --- 3. Execution Loop ---
print(f"Starting batch generation: {len(work_queue)} videos.")

# Updated CSV Header to include VRAM
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["prompt_index", "pattern", "num_frames", "duration_seconds", "peak_vram_gb", "status"])

for job in work_queue:
    idx = job["prompt_idx"]
    pattern = job["pattern"]
    frames = job["frames"]
    prompt_text = job["prompt"]
    
    output_file = f"{output_dir}/{pattern}_p{idx}_f{frames}.mp4"
    print(f"\n--- Prompt {idx+1}/{num_prompts} | Frames: {frames} | Pattern: {pattern} ---")

    # Clear memory and reset stats BEFORE generation
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    status = "Success"
    peak_vram = 0.0

    try:
        # Apply patch if pattern is radial
        if pattern == "radial":
            replace_wan_attention(
                pipe,
                height,
                width,
                frames, 
                1,      # dense_layers
                12,      # dense_timesteps
                0.2,    # decay_factor
                pattern,
                True    # use_sage_attention
            )
        
        # Inference
        with torch.no_grad():
            output = pipe(
                prompt=prompt_text,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            ).frames[0]
        
        # Capture Peak VRAM IMMEDIATELY after inference
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_vram = peak_bytes / (1024 ** 3) # Convert to GB
            
        export_to_video(output, output_file, fps=16)

    except Exception as e:
        print(f"Error: {e}")
        status = f"Failed: {e}"

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f}s | Peak VRAM: {peak_vram:.2f} GB")
    
    # Log to CSV
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([idx, pattern, frames, round(elapsed, 2), round(peak_vram, 2), status])

print(f"\nBatch generation complete. Metrics saved to {csv_filename}")