### Squeeze - and - Excitation Networks implementation (pytorch)

- For training

  ~~~
  python main.py 
  # argparser Default 
  --print_freq 32 --save_dir ./save_model/ --save_every 10 --lr 0.1 --weight_decay 1e-4 --momentum 0.9 --Epoch 80 --batch_size 128 --test_batch_size 100 --cutout True --n_masks 1 --length 16
  ~~~
  
- **Result** 

|               | resnet(from yeonsu repository) | SE + resnset (this code) + batchnorm | SEresnet + cutout + batch norm (this code) (cut length : 16) |
| ------------- | :----------------------------: | :----------------------: | :---------------------------------------------: |
| top - 1 error |              6.27              |           6.15           |                      4.76                       |

- Group normalization 

|               | SE + resnset (this code) + groupnorm | SEresnet + cutout + groupnorm (this code) (cut length : 16) |
| ------------- | :----------------------: | :---------------------------------------------: |
| top - 1 error |                     |                                    |

---

##### Preprocessing

- **Data augmentation**
  - 4pixels padded
  - Randomly 32 x 32 crop
  - Horizontal Flip
  - Normalization with mean and standard deviation
  - Option
    - Cutout  - Masking part to zero

##### Parameters

- Weight_initialization - kaiming_normal
- Optimizer
  - SGD
    - Learning_rate : 0.1
    - Milestones [250, 375]
    - gamma : 0.1
  - Weight_decay : 0.4
  - momentum : 0.9

##### Others

- group normalization 
- weight standardization (in progress...)
- Shake-Shake regularization (in progress...)
