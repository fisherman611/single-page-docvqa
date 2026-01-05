python models/multimodal_retriever/stage3_multimodal_retriever/train_multimodal_biencoder.py \
    --data data/augmented_data/augmented_train_spdocvqa.json \
    --text_model openai/clip-vit-base-patch32 \
    --image_model openai/clip-vit-base-patch32 \
    --epochs 20 \
    --hard_mining_every 2 \
    --hard_topm 50 \
    --hard_neg_k 3 \
    --query_view full \
    --index_type flat