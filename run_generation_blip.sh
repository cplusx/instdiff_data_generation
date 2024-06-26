LAION_FOLDER="/home/ubuntu/cjiaxin_16T/LAION400M512/laion400m-images"
folder_idx_start=1000
folder_idx_end=1100

run_generation() {
    cuda_device=$1
    folder_idx=$2
    CUDA_VISIBLE_DEVICES=${cuda_device} python automatic_label_blip2.py \
        --num_jobs 1 \
        --config config/GroundingDINO_SwinT_OGC.py \
        --ram_checkpoint models/ram_swin_large_14m.pth \
        --grounded_checkpoint models/groundingdino_swint_ogc.pth \
        --sam_checkpoint models/sam_vit_h_4b8939.pth \
        --box_threshold 0.25 \
        --text_threshold 0.2 \
        --iou_threshold 0.5 \
        --train_data_path laion_process_json_512/process_${folder_idx}.json \
        --output_dir "generated_data_blip/batch_${folder_idx}" \
        --laion_folder "$LAION_FOLDER"
}

# run_generation 0,1,2,3,4,5,6,7 0

current_folder_idx=$folder_idx_start

while true; do
    # get the device id of the first available GPU
    CUDA_DEVICE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | awk '$1==0 {print NR-1; exit}' | head -n 1)
    # if a GPU is available, run the command
    if [ -n "$CUDA_DEVICE" ]; then
        echo "Running command on GPU $CUDA_DEVICE"
        run_generation "$CUDA_DEVICE" "$current_folder_idx" &
        current_folder_idx=$((current_folder_idx+1))
        sleep 60
    else
        # no GPU is available, sleep for a minute
        echo "No GPU available, sleeping for a minute..."
        sleep 60
    fi
    if [ "$current_folder_idx" -ge "$folder_idx_end" ]; then
        wait # Wait for all background processes to finish
        break
    fi
done
