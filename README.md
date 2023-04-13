# pylot
### Reinforcement Learning for Flight Simulation
**Tanay Biradar, Conner Khudaverdyan, Casey Linden, Edison Zhang**

<p align="center">
    <img src="./demos/pylot-video-demo.gif" alt="Pylot Demo" width="50%"/>
</p>


## Setup

Create a file `.env` with the following environment variables:

```
WANDB_API_KEY=your_api_key
WANDB_ENTITY=your_wandb_username
TOTAL_TIMESTEPS=timesteps_to_train_for
```

- Training: Run `docker-compose build` to build the image and `docker-compose up` to train.
    The model will be saved to `runs`
    - To train with GPU, run `docker-compose` with `docker-compose-cuda.yml`
- Inference: On your local machine, run: `python ppo_inference.py --model-path="path-to-cleanrl_model-file"`
    - NOTE: [FlightGear](http://flightgear.org/) must be installed for
      visualization.  Ensure that `fgfs --version` runs properly

## Credits
- [CleanRL](https://github.com/vwxyzjn/cleanrl) for baseline [PPO](https://openai.com/research/openai-baselines-ppo) implementation
- [gym_jsbsim](https://github.com/Gor-Ren/gym-jsbsim) for Gym + JSBSim hooks and environment reward function
- [jsbsim](https://github.com/JSBSim-Team/jsbsim)
- Richter, D.J., & Calix, R.A. (2021). QPlane: An Open-Source Reinforcement Learning Toolkit for Autonomous Fixed Wing Aircraft Simulation. Proceedings of the 12th ACM Multimedia Systems Conference.
- Note: Everything in `utils` is not our own code but has been modified to fit
  the Dockerization needs of this repo. The resulting Docker images are thus self-contained.
