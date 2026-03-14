## Path Planning

Path planning allows to build a trajectory for the robot to move from its current configuration to a target position, without colliding with itself or the remaining of the objects in the scene.

2 options are available:

- `GenesisPathPlanner` – Genesis built-in collision-free plan_path, supports multi-environment natively
- `OmplPathPlanner` – relying on OMPL library

## OMPL

### Build

Build the library from source.


```
git clone https://github.com/ompl/ompl
cd ompl
export OMPL_HOME=`pwd`
cd build/Release
cmake ../.. -DOMPL_BUILD_PYTHON_BINDINGS=ON
make -j$(nproc)
```

To run the script, specify the custom location of the wheel built from source,

```
PYTHONPATH=$OMPL_HOME/build/Release/nanobinds python scripts/sim/run_sim_policy.py
```