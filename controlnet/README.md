I use kaggle:
step 1: 
!git clone https://github.com/BFCmath/DreamBooth.git
%cd DreamBooth
step 2: (manually check if extract pose work)
!python extract_pose.py --input_dir ./dataset --output_dir ./data --instance_prompt "a photo of sks person"
step 3: check normal without dreambooth, what is the output
!python infer_controlnet.py \
    --prompt "a photo of person wearing jacket, high quality" \
    --input_image ./data/conditioning/001.png \
    --controlnet_model lllyasviel/control_v11p_sd15_openpose \
    --detector none \
    --num_images 2 \
    --output_dir ./test_pose_output 
step 4: train 
!python dreambooth_controlnet.py \
    --data_dir ./data \
    --output_dir ./output/controlnet-openpose-dreambooth \
    --instance_prompt "a photo of sks person" \
    --class_prompt "a photo of person" \
    --with_prior_preservation \
    --num_class_images 100 \
    --sample_batch_size 8 \
    --max_train_steps 400 \
    --learning_rate 1e-6 \
    --repeats 1 \
    --num_class_images 20

step 5: infer finetuned model
# Inference with trained ControlNet (after training)
!python infer_controlnet.py \
    --prompt "a photo of sks person" \
    --input_image ./data/conditioning/001.png \
    --controlnet_model ./output/controlnet-openpose-dreambooth \
    --detector none \
    --num_images 4 \
    --seed 42 \
    --output_dir ./inference_output