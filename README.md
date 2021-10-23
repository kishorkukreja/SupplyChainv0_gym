# SupplyChainv0-gym
## Reinforcement Learning for Optimal inventory policy
## Environments for Supply chain

I sincerely thank Christian D. Hubbs and Hector D. Perez and Owais Sarwar and Nikolaos V. Sahinidis and Ignacio E. Grossmann and John M. Wassick for their work on Multi-period inventory and network management.
The enviroments have been borrowed from OR-gym library and have been modified to suit the needs
SupplychainV0 has been borrowed from Kevin Greevers CBC case study on reinforcement learning for inventory optimization

This library contains environments consisting of operations research problems which adhere to the OpenAI Gym API. The purpose is to bring reinforcement learning to the operations research community via accessible simulation environments featuring classic problems that are solved both with reinforcement learning as well as traditional OR techniques.

## Installation

This library requires Python 3.5+ in order to function.

For the RL algorithms, Ray 1.0.0 is required.

You can install directly from GitHub with:

```
git clone https://github.com/kishorkukreja/SupplyChainv0_gym.git
cd SupplyChain-gym
pip install -e .
```

## References
```
@misc{HubbsOR-Gym,
    author={Christian D. Hubbs and Hector D. Perez and Owais Sarwar and Nikolaos V. Sahinidis and Ignacio E. Grossmann and John M. Wassick},
    title={OR-Gym: A Reinforcement Learning Library for Operations Research Problems},
    year={2020},
    Eprint={arXiv:2008.06319}
}
```

## Environments
- `SupplyChainEnv-v0`: multi-echelon supply chain for Linear, divergent and general Supply chain.
- `InventoryManagement-v1`: multi-echelon supply chain re-order problem with backlogs.
- `InventoryManagement-v2`: multi-echelon supply chain re-order problem without backlog.
- `NetworkManagement-v1`: multi-echelon supply chain network problem with backlogs from [Perez et al.](https://www.mdpi.com/2227-9717/9/1/102).
- `NetworkManagement-v2`: multi-echelon supply chain network problem without backlogs from [Perez et al.](https://www.mdpi.com/2227-9717/9/1/102).
