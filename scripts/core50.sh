CUDA_VISIBLE_DEVICES=2 python -m main.main --dataset=seq-core50 --model=udil --lr=5e-5 \
    --n-epochs=100 --batch-size=128 --buffer-size=500 --buffer-batch-size=500 --backbone=resnet18 \
    --discriminator=mnistmlp --disc-num-layers=4 --loss=ce --disc-k=1 --disc-lr=1e-5 \
    --task-weight-k=1 --task-weight-lr=2e-3 --loop-k=1 --epoch-scaling=const --C=5 \
    --visualize --checkpoint --encoder-lambda=0.1 --num-workers=8 --encoder-mu=10 --opt=adam \
    --loss-form=sum --seed=1208 --supcon-lambda=0.1 --supcon-temperature=0.07 --kd-threshold=2 \
    --supcon-sim=l2 --supcon-first-domain --wandb-name=Core50-UDIL