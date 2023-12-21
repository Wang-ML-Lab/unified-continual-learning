CUDA_VISIBLE_DEVICES=1 python -m main.main --dataset=rot-mnist --model=udil --lr=1e-5 \
    --n-epochs=10 --batch-size=128 --buffer-size=400 --buffer-batch-size=400 --backbone=mnistmlp \
    --discriminator=mnistmlp --disc-num-layers=4 --loss=ce --disc-k=1 --disc-lr=1e-5 \
    --task-weight-k=1 --task-weight-lr=2e-3 --loop-k=1 --epoch-scaling=const --C=100 \
    --visualize --checkpoint --encoder-lambda=0.5 --num-workers=8 --encoder-mu=1 \
    --opt=adam --loss-form=sum --seed=1208 --supcon-lambda=0.01 --supcon-temperature=0.07 \
    --kd-threshold=1 --supcon-sim=l2 --supcon-first-domain --wandb-name=RotMNIST-UDIL