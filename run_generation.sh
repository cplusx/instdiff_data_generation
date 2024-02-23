LAION_FOLDER="/home/ubuntu/cjiaxin_16T/LAION400M/laion400m-images"
folder_idx=1
python automatic_label_qwen.py \
    --num_jobs 1 \
    --config config/GroundingDINO_SwinT_OGC.py \
    --ram_checkpoint models/ram_swin_large_14m.pth \
    --grounded_checkpoint models/groundingdino_swint_ogc.pth \
    --sam_checkpoint models/sam_vit_h_4b8939.pth \
    --box_threshold 0.25 \
    --text_threshold 0.2 \
    --iou_threshold 0.5 \
    --train_data_path laion_process_json/process_${folder_idx}.json \
    --output_dir "generated_data/batch_${folder_idx}" \
    --laion_folder $LAION_FOLDER \
    # --sam_hq_checkpoint Grounded-Segment-Anything/sam_hq_vit_h.pth \
    # --use_sam_hq \
    # --device "cuda" \
