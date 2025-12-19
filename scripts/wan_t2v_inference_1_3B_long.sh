# this is the setting for 1x length T2V inference
dense_layers=2
dense_timesteps=2

prompt=$(cat examples/long_prompt_2.txt)

python inference_scripts/wan_t2v_inference.py \
    --model_id "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
    --prompt "$prompt" \
    --height 480 \
    --width 640 \
    --num_frames 241 \
    --dense_layers $dense_layers \
    --dense_timesteps $dense_timesteps \
    --decay_factor 0.2 \
    --pattern "radial" \
    --use_sage_attention \
    --output_file "radial_long.mp4"
    



# this is the setting for 2x length T2V inference
# dense_layers=2
# dense_timesteps=2

# python wan_t2v_inference.py \
#     --prompt "$prompt" \
#     --height 720 \
#     --width 1280 \
#     --num_frames 161 \
#     --pattern "radial" \
#     --dense_layers $dense_layers \
#     --dense_timesteps $dense_timesteps \
