# FedCon
If you have any other questions, you can contact 3061048858@qq.com Thank you
## INTRODUCTION
This is the Pytorch code of the paper entitled "A Model Consistency-based Mechanism to Safeguard Federated Learning in Vulnerable and Heterogeneous IoT Environments".  
  
The diri_distribution records the data distribution, and the models stores the neural network models used in the article，fedcon_learner controls the server side, fedcon_client controls the client side, fedcon_main starts the global training process.


##  THE PROPOSED MECHANISM FEDCON
We focus on defending against data poisoning attacks in FL. Our overall framework is presented in Figure 1, and the specific process of malicious model detection is shown in Figure 2.
<div align="center">
  <img src="https://github.com/IntelligentSystemsLab/FedCon/blob/main/img/FedCon_overview.png">
</div>
Figure 1. The overview of FedCon consisting of A) Local Training Phase, B) Malicious Model Identification, and C) Global Aggregation Phase.

<div align="center">
  <img src="https://github.com/IntelligentSystemsLab/FedCon/blob/main/img/FedCon_detail.png">
</div>
