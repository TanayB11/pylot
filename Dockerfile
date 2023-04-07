FROM ubuntu:latest

RUN mkdir -p /pylot
WORKDIR /pylot

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y cmake

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install numpy gym==0.21.0 tensorboard jsbsim cython wandb

COPY . .

WORKDIR /pylot/utils/gym-jsbsim
RUN pip3 install .

# build jsbsim from scratch
RUN mkdir /pylot/utils/jsbsim-code/build
WORKDIR /pylot/utils/jsbsim-code/build
RUN cmake .. && make

WORKDIR /pylot/src

# ARG WANDB_ENTITY
# ARG TOTAL_TIMESTEPS
# CMD python3 ppo_continuous_action.py --track 1 --wandb-entity ${WANDB_ENTITY} --total-timesteps ${TOTAL_TIMESTEPS}
