###Squeeze - and - Excitation Networks implementation (pytorch)

- For training

  ~~~
  python main.py 
  # argparser Default 
  --print_freq 32 --save_dir ./save_model/ --save_every 10 --lr 0.1 --weight_decay 1e-4 --momentum 0.9 --Epoch 80 --batch_size 128 --test_batch_size 100 --cutout True --n_masks 1 --length 16
  ~~~
  
- **Result** 

|               | resnet(from yeonsu repository) | SE + resnset (this code) | SEresnet + cutout (this code) (cut length : 16) |
| ------------- | :----------------------------: | :----------------------: | :---------------------------------------------: |
| top - 1 error |              6.27              |           6.15           |                      4.76                       |

---
