# Covariance-Temporal-GCN-for-Traffic-Forecasting

This work presents a novel approach for traffic foreasting, that leverages covariance based temporal embeddings in the data to create a graph filter to prune influential nodes from previous timestep. This is combined with graph convolutions to enhance spatio-temporal learning. 

## Results

Accuracy                   | RMSE 
:-------------------------:|:-------------------------:
![](images/Accuracy.png)  |  ![](images/RMSE.png)


TGCN - SZ-Taxi             | cVTGCN - SZ-Taxi
:-------------------------:|:-------------------------:
![](images/lostgcn.png)  |  ![](images/loscVtgcn.png)

TGCN - Los-Loop            | cVTGCN - Los-Loop
:-------------------------:|:-------------------------:
![](images/tgcn.png)  |  ![](images/cvtgcn.png)



## Built With

* [PyTorch](https://pytorch.org/) - Deep Learning Framework
* [SpatioTemporal Neural Networks](https://github.com/andrea-cavallo-98/STVNN)
* [TGCN](https://huggingface.co/](https://github.com/lehaifeng/T-GCN/tree/master)


## Authors
- [Sai Vamsi Alisetti](https://github.com/Vamsi995)
- [Vikas Kalagi](https://github.com/vikaskalagi)
