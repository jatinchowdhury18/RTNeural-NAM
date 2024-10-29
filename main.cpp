#include <NAM/dsp.h>
#include <RTNeural/RTNeural.h>
#include <iostream>

#include "wavenet/wavenet_model.hpp"

int main()
{
    const auto model_path { std::string { ROOT_DIR } + "OB1 Mesa DC-5 PM.nam" };

    std::cout << "Loading model from path: " << model_path << std::endl;

    nam::dspData model_data;
    auto nam_dsp = nam::get_dsp (model_path, model_data);

    wavenet::Wavenet_Model<float,
                           RTNeural::DefaultMathsProvider,
                           wavenet::Layer_Array<float, 1, 1, 8, 16, 3, false, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512>,
                           wavenet::Layer_Array<float, 16, 1, 1, 8, 3, true, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512>>
        rtneural_wavenet;
    rtneural_wavenet.load_weights (model_data.config, model_data.weights);

    nam_dsp->prewarm();
    std::cout << std::endl;

    static constexpr size_t N = 2048;
    std::vector<double> input;
    input.resize (N, 0.0);
    std::vector<double> output_nam;
    output_nam.resize (N, 0.0);
    std::vector<float> output_rtneural;
    output_rtneural.resize (N, 0.0);

    for (size_t n = 0; n < input.size(); ++n)
        input[n] = std::sin (3.14 * static_cast<double> (n) * 0.01);

    auto start = std::chrono::high_resolution_clock::now();
    nam_dsp->process (input.data(), output_nam.data(), N);
    auto end = std::chrono::high_resolution_clock::now();
    const auto duration_nam = std::chrono::duration_cast<std::chrono::duration<double>> (end - start).count();
    std::cout << "NAM: " << duration_nam << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (size_t n = 0; n < input.size(); ++n)
    {
        // nam_dsp->process (input.data() + n, output_nam.data() + n, 1);
        output_rtneural[n] = rtneural_wavenet.forward (static_cast<float> (input[n]));
    }
    end = std::chrono::high_resolution_clock::now();
    const auto duration_rtneural = std::chrono::duration_cast<std::chrono::duration<double>> (end - start).count();
    std::cout << "RTNeural: " << duration_rtneural << std::endl;

    std::cout << "RTNeural is: " << duration_nam / duration_rtneural << "x faster" << std::endl;

    return 0;
}
