# RTNeural-NAM

Implementation of a [Neural Amp Modeler](https://github.com/sdatkinson/NeuralAmpModelerCore)-style Wavenet
using [RTNeural](https://github.com/jatinchowdhury18/RTNeural).

## Setup

```bash
# Clone repo and dependencies
git clone https://github.com/jatinchowdhury18/RTNeural-NAM
cd RTNeural-NAM
git clone https://github.com/jatinchowdhury18/RTNeural
git clone https://github.com/sdatkinson/NeuralAmpModelerCore

# Configure and build with CMake
cmake -Bbuild -G<generator>
cmake --build build --target RTNeural-NAM --parallel
```

## TODO:
- Dense layer with no bias
- Better way to handle dilations (`std::integer_sequence`)
- Optimize
- Gated activations
- Head bias
