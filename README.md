# RTNeural-NAM

Implementation of a [Neural Amp Modeler](https://github.com/sdatkinson/NeuralAmpModelerCore)-style Wavenet
using [RTNeural](https://github.com/jatinchowdhury18/RTNeural).

## Setup

```bash
# Clone repo and dependencies
git clone https://github.com/jatinchowdhury18/RTNeural-NAM
cd RTNeural-NAM
git submodule update --init --recursive

# Configure and build with CMake
cmake -Bbuild -G<generator>
cmake --build build --target RTNeural_NAM --parallel
```

## TODO:
- More optimizations
- Gated activations
