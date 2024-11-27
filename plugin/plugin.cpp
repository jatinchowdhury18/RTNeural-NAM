#include "plugin.h"
#include <chowdsp_gui/chowdsp_gui.h>

void Plugin::prepareToPlay (double sample_rate, int samples_per_block)
{
    const auto model_path { std::string { ROOT_DIR } + "OB1 Mesa DC-5 PM.nam" };
    nam_dsp = nam::get_dsp (model_path);

    nlohmann::json model_json {};
    std::ifstream { model_path, std::ifstream::binary } >> model_json;
    rtneural_wavenet.load_weights (model_json);

    nam_dsp->prewarm();
    rtneural_wavenet.prepare (samples_per_block);
    rtneural_wavenet.prewarm();
}

void Plugin::processAudioBlock (juce::AudioBuffer<float>& buffer)
{
    const auto num_channels = buffer.getNumChannels();
    const auto num_samples = buffer.getNumSamples();

    auto* data = buffer.getWritePointer (0);
    const auto mode = state.params.mode->getIndex();
    if (mode == 0)
    {
        nam_dsp->process (data, data, num_samples);
    }
    else if (mode == 1)
    {
        for (int n = 0; n < num_samples; ++n)
            data[n] = rtneural_wavenet.forward (data[n]);
    }
    else
    {
        rtneural_wavenet.forward (data, data, num_samples);
    }

    for (int ch = 1; ch < num_channels; ++ch)
        buffer.copyFrom (ch, 0, buffer, 0, 0, num_samples);
}

juce::AudioProcessorEditor* Plugin::createEditor()
{
    return new chowdsp::ParametersViewEditor { *this };
}

// This creates new instances of the plugin
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new Plugin();
}
