#include <NAM/dsp.h>
#include <RTNeural/RTNeural.h>
#include <iostream>
#include <math_approx/math_approx.hpp>

#include "wavenet/wavenet_model.hpp"

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

std::vector<float> generate_test_signal (size_t N)
{
    std::vector<float> signal;
    signal.resize (N, 0.0f);

    for (size_t n = 0; n < N; ++n)
        signal[n] = std::sin (3.14f * static_cast<float> (n) * 0.01f);

    return signal;
}

void compute_error (std::span<const float> output_nam, std::span<const float> output_rtneural)
{
    const auto N = output_nam.size();
    float error_sq_accum = 0.0f;
    for (size_t n = 0; n < N; ++n)
    {
        const auto err = output_nam[n] - output_rtneural[n];
        error_sq_accum += err * err;
    }
    const auto rms_error = std::sqrt (error_sq_accum / static_cast<float> (N));
    std::cout << "RMS error: " << rms_error << std::endl;
}

template <typename Model>
void load_model (Model& model, const std::string& model_path)
{
    nlohmann::json model_json {};
    std::ifstream { model_path, std::ifstream::binary } >> model_json;
    model.load_weights (model_json);
}

void test_ob1_model()
{
    const auto model_path { std::string { ROOT_DIR } + "OB1 Mesa DC-5 PM.nam" };
    std::cout << "Testing model: " << model_path << std::endl;

    auto nam_dsp = nam::get_dsp (model_path);

    using Dilations = wavenet::Dilations<1, 2, 4, 8, 16, 32, 64, 128, 256, 512>;
    wavenet::Wavenet_Model<float,
                           1,
                           wavenet::Layer_Array<float, 1, 1, 8, 16, 3, Dilations, false, NAMMathsProvider>,
                           wavenet::Layer_Array<float, 16, 1, 1, 8, 3, Dilations, true, NAMMathsProvider>>
        rtneural_wavenet;
    load_model (rtneural_wavenet, model_path);

    nam_dsp->prewarm();
    rtneural_wavenet.prewarm();

    static constexpr size_t N = 2048;
    auto input = generate_test_signal (N);
    std::vector<float> output_nam;
    output_nam.resize (N, 0.0f);
    std::vector<float> output_rtneural;
    output_rtneural.resize (N, 0.0f);

    rtneural_wavenet.prepare (N);

    nam_dsp->process (input.data(), output_nam.data(), N);
    rtneural_wavenet.forward (input.data(), output_rtneural.data(), N);
    // Un-comment this to test per-sample processing
    // for (size_t n = 0; n < N; ++n)
    //     output_rtneural[n] = rtneural_wavenet.forward (input[n]);

    compute_error (output_nam, output_rtneural);
}

template <typename F1, typename F2>
auto time_it (F1&& f1, F2&& f2)
{
    auto start = std::chrono::high_resolution_clock::now();
    f1();
    auto end = std::chrono::high_resolution_clock::now();
    const auto dur1 = std::chrono::duration_cast<std::chrono::duration<double>> (end - start).count();

    start = std::chrono::high_resolution_clock::now();
    f2();
    end = std::chrono::high_resolution_clock::now();
    const auto dur2 = std::chrono::duration_cast<std::chrono::duration<double>> (end - start).count();

    return std::make_pair (dur1, dur2);
}

void bench_ob1_model (size_t N, size_t block_size)
{
    const auto model_path { std::string { ROOT_DIR } + "OB1 Mesa DC-5 PM.nam" };
    std::cout << "Benchmarking model: " << model_path << std::endl;

    nam::dspData model_data;
    auto nam_dsp = nam::get_dsp (model_path, model_data);

    using Dilations = wavenet::Dilations<1, 2, 4, 8, 16, 32, 64, 128, 256, 512>;
    wavenet::Wavenet_Model<float,
                           1,
                           wavenet::Layer_Array<float, 1, 1, 8, 16, 3, Dilations, false, NAMMathsProvider>,
                           wavenet::Layer_Array<float, 16, 1, 1, 8, 3, Dilations, true, NAMMathsProvider>>
        rtneural_wavenet;
    load_model (rtneural_wavenet, model_path);
    rtneural_wavenet.prepare ((int) block_size);

    nam_dsp->prewarm();
    rtneural_wavenet.prewarm();

    auto input = generate_test_signal (N);
    std::vector<float> output;
    output.resize (N, 0.0f);

    size_t n_blocks = N / block_size;

    const auto [duration_nam, duration_rtneural] = time_it (
        [&]
        {
            size_t offset = 0;
            for (int buf = 0; buf < n_blocks; buf++)
            {
                nam_dsp->process (input.data() + offset, output.data() + offset, (int) block_size);
            }
        },
        [&]
        {
            size_t offset = 0;
            for (int block = 0; block < n_blocks; block++)
            {
                const auto* in_data = input.data() + offset;
                auto* out_data = output.data() + offset;

                rtneural_wavenet.forward (in_data, out_data, (int) block_size);

                // un-comment this to test per-sample processing
                // for (int n = 0; n < block_size; ++n)
                //     out_data[n] = rtneural_wavenet.forward (in_data[n]);

                offset += block_size;
            }
        });

    std::cout << "NAM: " << duration_nam << std::endl;
    std::cout << "RTNeural: " << duration_rtneural << std::endl;
    std::cout << "RTNeural is: " << duration_nam / duration_rtneural << "x faster" << std::endl;
}

int main()
{
    nam::activations::Activation::enable_fast_tanh();

    test_ob1_model();
    bench_ob1_model (1 << 15, 128);

    return 0;
}
