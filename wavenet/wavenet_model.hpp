#pragma once

#include "wavenet_layer_array.hpp"

namespace wavenet
{
template <typename T,
          int condition_size,
          typename MathsProvider = RTNeural::DefaultMathsProvider,
          typename... LayerArrays>
struct Wavenet_Model
{
    std::tuple<LayerArrays...> layer_arrays;
    Eigen::Matrix<T, 16, 1> head_input {};
    T head_scale = (T) 0;

    void load_weights (const nlohmann::json& model_config, std::vector<float>& model_weights)
    {
        auto weights_iterator = model_weights.begin();
        RTNeural::modelt_detail::forEachInTuple (
            [&weights_iterator] (auto& layer, size_t)
            {
                layer.load_weights (weights_iterator);
            },
            layer_arrays);

        head_scale = *weights_iterator++;

        // Make sure we use the all of the weights exactly
        assert (std::distance (model_weights.begin(), weights_iterator) == model_weights.size());
    }

    T forward (T input) noexcept
    {
        RTNeural::modelt_detail::forEachInTuple (
                [this, input] (auto& layer_array, auto index_t)
                {
                    static constexpr size_t index = index_t;
                    if constexpr (index == 0)
                    {
                        const auto v_ins = Eigen::Matrix<T, 1, 1>::Constant (input);
                        head_input.setZero();
                        Eigen::Map<Eigen::Matrix<T, 16, 1>, RTNeural::RTNeuralEigenAlignment> head_input_map { head_input.data() };
                        std::get<0> (layer_arrays).forward (v_ins, Eigen::Matrix<T, condition_size, 1>::Zero(), head_input_map);
                    }
                    else
                    {
                        std::get<index> (layer_arrays).forward (std::get<index - 1> (layer_arrays).layer_outputs,
                                                                Eigen::Matrix<T, condition_size, 1>::Zero(),
                                                                std::get<index - 1> (layer_arrays).head_outputs);
                    }
                },
                layer_arrays);



        return std::get<std::tuple_size_v<decltype (layer_arrays)> - 1> (layer_arrays).head_outputs[0] * head_scale;
    }
};
} // namespace wavenet
