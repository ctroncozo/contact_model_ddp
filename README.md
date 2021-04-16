# Trajectory Optimization With Implicit Hard Contacts
<p>
Paper Reference: https://ieeexplore.ieee.org/document/8403260

* Use the technique from the paper reference to optimize gait sequence, timing, and whole-body motion by delegating the contact constraints to the system dynamics. 

* The method attempts to incorporate the complementary constraints into the OC problem. This approach is capable of finding dynamic movements without specifying the contact sequence.

#### Goal of this implementation:
* Replace the action model from crocoddyl by the time stepping algorithm used in the reference paper.
* To do that the **DifferentialActionModelContactFwdDynamics** and **IntegratedActionModelEuler** from crocoddyl needs to be reimplemted such that isntead is used the time-stepping algorithm
```
def create_walking_problem():
  1. Create swing_feep_model()
     * For each fee create a model for the swing phase
     * 
 ```
</p>
  



<p align="center">

<img src="./files/time_step_algo.png" width="350" height="430" title="Algorithm" />
</p>                                                                  
