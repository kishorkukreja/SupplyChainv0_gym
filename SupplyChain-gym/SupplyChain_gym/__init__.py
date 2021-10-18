from gym.envs.registration import register

register(
    id="SupplyChainEnv-v0",
    entry_point="SupplyChain_gym.envs:InventoryEnv",
)

register(
    id="NetworkManagement-v1",
    entry_point="SupplyChain_gym.envs:NetInvMgmtBacklogEnv",
)

register(
    id="NetworkManagement-v2",
    entry_point="SupplyChain_gym.envs:NetInvMgmtLostSalesEnv",
)
