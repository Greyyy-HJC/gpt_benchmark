# This is the repo to collect some benchmark of gpt usage

## Contents

- [conf](./conf/): generated configurations for benchmark, so far all the configurations are pure gauge configurations.
    - [S8T8](./conf/S8T8/): means 8 lattice units in the spatial directions and 8 lattice units in the time direction
    - [S16T8](./conf/S16T8/): means 16 lattice units in the spatial directions and 8 lattice units in the time direction

- [conf_gen](./conf_gen/): scripts to generate configurations
    - [pure_gauge_wilson](./conf_gen/pure_gauge_wilson.py): generate pure gauge configurations with Wilson gauge action

- [dump](./dump/): dumped files, dump via `gvar.dump`, load via `gvar.load`

- [gfix](./gfix/): scripts for gauge fixing configurations

- [meas](./meas/): scripts for measurements