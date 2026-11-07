# Fine-Tune with LoRA

## Prediction (written before running the sweep)

I believe that LoRA rank 16 will have accuracy close to
full fine-tuning since it has more parameters to train on
and learn more flexible updates.

At the same time, I believe that LoRA rank 1 will be less
accurate than LoRA rank 16 and even full fine-tuning due to
its lower rank, which reduces the learning capacity.

Nevertheless, this is only my prediction, and the real
accuracy may differ from the one described above.
