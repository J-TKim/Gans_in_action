#!/usr/bin/env python
# coding: utf-8

# <table align="left">
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/J-TKim/Gans_in_action/blob/master/Ch8/Ch9_CycleGAN.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩에서 실행하기</a>
#   </td>
# </table>

# In[1]:


# 코드 9-1 패키지 임포트
from __future__ import print_function, division
import scipy
from tensorflow.keras.datasets import mnist
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os


# In[2]:


# 코드 9-2 CycleGAN 클래스
class CycleGAN():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        # 데이터 로더 설정
        self.dataset_name = "apple2orange"
        # DataLoader 객체를 사용해 전처리된 데이터를 임포트합니다.
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))
        
        # D(PatchGAN)의 출력 크기를 계산합니다.
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch + 1)
        
        # G의 첫 번째 층에 있는 필터의 개수
        self.gf = 32
        # D의 첫 번째 층에 있는 필터의 개수
        self.df = 64
        
        # 사이클-일관성 손실 가중치
        self.lambda_cycle = 10.0
        # 동일성 손실 가중치
        self.lambda_id = 0.9 * self.lambda_cycle
        
        optimizer = Adam(0.0002, 0.5)
        
        # 판별자를 만들고 컴파일합니다.
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss="mse",
                         optimzier=optimizer,
                         metrics=["accuracy"])
        self.d_B.compile(loss="mse",
                         optimzier=optimizer,
                         metrics=["accuracy"])
        
        # 여기서부터 생성자의 계산 그래프를 만듭니다. 처음 두 라인이 생성자를 만듭니다.
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()
        
        # 두 도메인의 입력 이미지
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        
        # 이미지를 다른 도메인으로 변환합니다.
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        
        # 원본 도메인으로 이미지를 다시 변환합니다.
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        
        # 동일한 이미지 매핑
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)
        
        # 연결 모델에서는 생성자만 훈련 합니다.
        self.d_A.trainable = False
        self.d_B.trainable = False
        
        # 판별자가 변환된 이미지의 유효성을 결정합니다.
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)
        
        # 연결 모델은 판별자를 속이기 위한 생성자를 훈련합니다.
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       img_A_id, img_B_id])
        self.combined.compile(loss=["mse", "mse",
                                    "mae", "mae",
                                    "mae", "mae"],
                              loss_weight=[1, 1,
                                           self.lambda_cycle, self.lambda_cycle,
                                           self.lambda_id, self.lambda_id],
                              optimizer=optimizer)  


# In[3]:


def CycleGAN(CycleGAN):
    @staticmethod
    def conv2d(layer_input, filters, f_size=4, normalization=True):
        "다운샘플링 하는 동안 사용되는 층"
        d = Conv2D(filters, kernel_size=f_size,
                   strides=2, padding="same")(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d
    @staticmethod
    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        "업샘플링하는 동안 사용되는 층"
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1,
                   padding="same", activation="relu")(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u


# In[4]:


def CycleGAN(CycleGAN):
    def build_generator(self):
        "U-Net 생성자"
        
        # 이미지 입력
        d0 = Input(shape=self.img_shape)
        
        # 다운샘플링
        d1 = self.conv2d(d0, self.gf)
        d2 = self.conv2d(d1, self.gf * 2)
        d3 = self.conv2d(d2, self.gf * 4)
        d4 = self.conv2d(d3, self.gf * 8)
        
        # 업샘플링
        d5 = self.deconv2d(d4, d3, self.gf * 4)
        d6 = self.deconv2d(d5, d2, self.gf * 2)
        d7 = self.deconv2d(d6, d1, self.gf)
        
        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4,
                            strides=1, padding="same", activation="tanh")(u4)
        
        return Model(d0, output_img)


# In[5]:


def CycleGAN(CycleGAN):
    img = Input(shape=self.img_shape)
    
    d1 = self.conv2d(img, self.df, normalization=False)
    d2 = self.conv2d(img, self.df * 2)
    d3 = self.conv2d(img, self.df * 4)
    d4 = self.conv2d(img, self.df * 8)
    
    validity = Conv2D(1, kernel_size=4, strides=1, padding="same")(d4)
    
    return Model(img, validity)


# In[ ]:




