# FedCon
If you have any other questions, you can contact 3061048858@qq.com Thank you!
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
  <img src="https://github.com/IntelligentSystemsLab/FedCon/blob/main/img/FedCon_datail.png">
</div>
Figure 2. The detailed workflow of Malicious Clients Identification in FedCon, including A) Voting Matrix Calculation, and B) Model Voting Procedure.  

##  PAPER RESULTS
We conducted a holistic evaluation based on three standard datasets, FedCon can outperform five state-of-the-art baselines. Under the scenarios with high data heterogeneity level and malicious client rate, it can consistently maintain a stable and robust model performance as well as a low and rare attack success rate to secure the collaborative and privacy-preserving model learning process. 

<div align="center">
  <img src="https://github.com/IntelligentSystemsLab/FedCon/blob/main/img/FedCon_result.png">
</div>
