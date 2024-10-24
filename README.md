# PLEASSURE: Learning Physical Environment factors based on the Attention mechanism to Select Services for UsERs

The scope of the Internet of Things (IoT) environment has been expanding from private to public spaces, where selecting the most appropriate service by predicting the service quality has become a timely problem. 
However, IoT services relying on distributed devices to interact with the users can be physically affected by (1) uncertain environmental factors such as obstacles and (2) interference among services in the same environment. 
Using the traditional modeling-based approach, analyzing the influence of such factors on the service quality requires modeling efforts and lacks generalizability.
In this study, we propose _Learning Physical Environment factors based on the Attention mechanism to Select Services for UsERs (PLEASSURE)_, a novel framework that selects the most promising IoT service by learning the uncertain influence of environmental factors and predicting the long-term quality of each candidate service based on multi-agent reinforcement learning. 
Each service agent learns its quality prediction model solely from the users' feedback without additional information on the environment.
Furthermore, to capture the physical interference among services, we propose _fingerprint attention_ that summarizes the states of other service agents based on learnable fingerprint vectors.
We evaluate PLEASSURE by simulating various IoT environments with mobile users and IoT services.
The results show that PLEASSURE outperforms the baseline algorithms in rewards consisting of users' feedback on satisfaction and interference.
