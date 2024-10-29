#pragma once

#include "wavenet_layer_array.hpp"

namespace wavenet
{
template <typename T,
          typename MathsProvider = RTNeural::DefaultMathsProvider,
          typename... LayerArrays>
struct Wavenet_Model
{
    std::tuple<LayerArrays...> layer_arrays;
    Eigen::Matrix<T, 16, 1> head_io {};
    const T head_scale = (T) 0.02;

    void load_weights (const nlohmann::json& model_config, std::vector<float>& model_weights)
    {
        auto weights_begin = model_weights.begin();
        auto weights_iterator = model_weights.begin();

        RTNeural::modelt_detail::forEachInTuple (
            [&weights_iterator] (auto& layer, size_t)
            {
                layer.load_weights (weights_iterator);
            },
            layer_arrays);

            assert ((weights_iterator - weights_begin) == model_weights.size());    // Make sure we use the all of the weights exactly
            
            // std::cout << weights_json << std::endl;
    }

    T forward (T input) noexcept
    {
        head_io.setZero();

        const auto v_ins = Eigen::Matrix<T, 1, 1>::Constant (input);
        std::get<0> (layer_arrays).forward (v_ins, {}, head_io);

        return head_io[0] * head_scale;
    }
};
} // namespace wavenet
