#pragma once

#include <chowdsp_plugin_base/chowdsp_plugin_base.h>

#include <NAM/dsp.h>
#include <RTNeural/RTNeural.h>
#include <math_approx/math_approx.hpp>

#include "../wavenet/wavenet_model.hpp"

struct NAMMathsProvider
{
#if RTNEURAL_USE_EIGEN
    template <typename Matrix>
    static auto tanh (const Matrix& x)
    {
        // See: math_approx::tanh<3>
        const auto x_poly = x.array() * (1.0f + 0.183428244899f * x.array().square());
        return x_poly.array() * (x_poly.array().square() + 1.0f).array().rsqrt();
    }
#elif RTNEURAL_USE_XSIMD
    template <typename T>
    static T tanh (const T& x)
    {
        return math_approx::tanh<3> (x);
    }
#endif
};

struct Params : chowdsp::ParamHolder
{
    Params()
    {
        add (mode);
    }

    chowdsp::ChoiceParameter::Ptr mode {
        PID { "mode", 100 },
        "Mode",
        juce::StringArray { "NAM", "RTNeural (per-sample)", "RTNeural (block)" },
        0,
    };
};

struct Plugin : chowdsp::PluginBase<chowdsp::PluginStateImpl<Params>>
{
    Plugin() = default;

    void prepareToPlay (double sample_rate, int samples_per_block) override;
    void releaseResources() override {}
    void processAudioBlock (juce::AudioBuffer<float>& buffer) override;

    juce::AudioProcessorEditor* createEditor() override;

    std::unique_ptr<nam::DSP> nam_dsp {};
    using Dilations = wavenet::Dilations<1, 2, 4, 8, 16, 32, 64, 128, 256, 512>;
    wavenet::Wavenet_Model<float,
                           1,
                           wavenet::Layer_Array<float, 1, 1, 8, 16, 3, Dilations, false, NAMMathsProvider>,
                           wavenet::Layer_Array<float, 16, 1, 1, 8, 3, Dilations, true, NAMMathsProvider>>
        rtneural_wavenet;
};
