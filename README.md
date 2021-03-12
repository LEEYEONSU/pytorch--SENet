###Squeeze - and - Excitation Networks implementation (pytorch)

- For training

  ~~~
  python main.py 
  # argparser Default 
  --print_freq 32 --save_dir ./save_model/ --save_every 10 --lr 0.1 --weight_decay 1e-4 --momentum 0.9 --Epoch 80 --batch_size 128 --test_batch_size 100 --cutout True --n_masks 1 --length 4
  ~~~

---

- Dataset : CIFAR - 10

##### Network

- Squeeze and excitation (Attention)



##### Preprocessing

- **Data augmentation**

  - 4pixels padded

  - Randomly 32 x 32 crop 

  - Horizontal Flip

  - Normalization with mean and standard deviation

  - Option

    - Cutout - Masking part zero 
    - <img src="/Users/yeonsulee/Library/Application Support/typora-user-images/image-20210312142412454.png" alt="image-20210312142412454" style="zoom:33%;" />


##### Parameters

- Weight_initialization - kaiming_normal 
- Optimizer
  - SGD
    - Learning_rate : 0.1
    - Milestones [30, 80]
    - gamma : 0.1
  - Weight_decay : 0.4 
  - momentum : 0.9

##### Others

- Shake-Shake regularization (in progress...)

#### 

