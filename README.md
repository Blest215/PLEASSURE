# PLEASSURE: Learning Physical Environment factors based on the Attention mechanism to Select Services for UsERs

The scope of the Internet of Things (IoT) environment has been expanding from private to public spaces. 
With the increasing number of IoT services in public spaces, selecting the most appropriate service by predicting the quality of experience (QoE) has become a timely problem. 
However, IoT services that rely on physical devices can be affected by environmental factors such as obstacles and interference among services. 
Analyzing the influence of such factors on the QoE using the traditional model-based approach is costly and difficult to generalize.
In this study, we propose Learning Physical Environment factors based on the Attention mechanism to Select Services for UsERs (PLEASSURE), a novel method that selects the most promising IoT service by predicting the long-term QoE of the candidates. 
Each service agent learns its corresponding prediction model solely based on the users' feedback.
Furthermore, to capture the physical factors of the services, we present fingerprint attention that summarizes the states of the other service agents based on learnable fingerprints.
We evaluate PLEASSURE by simulating various IoT environments with mobile users and IoT services, and the results show that in most environments, PLEASSURE outperforms the benchmark algorithms in rewards consisting of users' feedback on satisfaction and interference.