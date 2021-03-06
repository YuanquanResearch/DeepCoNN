# DeepCoNN

This is our implementation for the paper:


*Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation.
In WSDM. ACM, 425-434.*


Two models:

1、DeepCoNN: This is the state-of-the-art method that uti-lizes deep learning technology to jointly model user and itemfrom textual reviews.

2、DeepCoNN++: We extend DeepCoNN by changing its share layer from FM to our neural prediction layer.


The two methods are used as the baselines of our method **NARRE** in the paper:


*Chong Chen, Min Zhang, Yiqun Liu, and Shaoping Ma. 2018. [Neural Attentional Rating Regression with Review-level Explanations.](http://www.thuir.cn/group/~YQLiu/publications/WWW2018_CC.pdf) 
In WWW'18.*

**Please cite our WWW'18 paper if you use our codes. Thanks!**

```
@inproceedings{chen2018neural,
  title={Neural Attentional Rating Regression with Review-level Explanations},
  author={Chen, Chong and Zhang, Min and Liu, Yiqun and Ma, Shaoping},
  booktitle={Proceedings of the 2018 World Wide Web Conference on World Wide Web},
  pages={1583--1592},
  year={2018},
}
```

Author: Chong Chen (cstchenc@163.com)

## Environments

- python 2.7
- Tensorflow (version: 0.12.1)
- numpy
- pandas


## Dataset

In our experiments, we use the datasets from  Amazon 5-core(http://jmcauley.ucsd.edu/data/amazon) and Yelp Challenge 2017(https://www.yelp.com/dataset_challenge).

## Example to run the codes		

Data preprocessing:

```
python loaddata.py	
python data_pro.py
```

Train and evaluate the model:

```
python train.py
```



Last Update Date: Jan 3, 2018



# 源泉修改
train.csv: user,item,rating
user_reviews: {userid: [review0, review1, ...]}  只是train中的，没有val/test中的
user_rid: {userid: [itemid1, itemid2, ...]}


u_text: {user: [wordid1, wordid2, ...]} review顺序无关，reviews内部有顺序
i_text: {item: [wordid1, wordid2, ...]}  两个text用的word_id映射可以不一样 奇怪

CDs: u_len/i_len 选0.8
Movies: u_len/i_len 选0.7

重要参数是drop_keep


deepconn:
music:
u_len, i_len: 2854 5208

CDs: 0.8 u_len的划分比例 (要么就直接选test，不在val上选)
参数：drop_keep
drop_keep: 0.8
(0.9035), 0.9860: /home/users/luyq/research/DeepCoNN/model/CDs_0.8_2017


Movies: 0.7 U_len的划分比例
drop_keep: 1.0
1.1469: /disk1t/lup/research/DeepCoNN/model/Movies_1.0.bak
[1.1528, 1.1454, 1.1471, 1.1424]

patient 由5变为3
DeepCoNN等待结果：CDs和Movies
/disk1t/lup/research/DeepCoNN/model