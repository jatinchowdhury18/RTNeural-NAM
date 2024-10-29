#pragma once

#include "wavenet_layer.hpp"

namespace wavenet
{
template <typename T,
          int in_size,
          int condition_size,
          int head_size,
          int channels,
          int kernel_size,
          bool has_head_bias,
          int... dilations>
struct Layer_Array
{
    RTNeural::DenseT<T, in_size, channels> rechannel; // no bias!
    std::tuple<Wavenet_Layer<T, condition_size, channels, kernel_size, dilations>...> layers;
    static constexpr auto num_layers = std::tuple_size_v<decltype(layers)>;
    RTNeural::DenseT<T, channels, head_size> head_rechannel; // head_bias = true
    decltype (RTNeural::DenseT<T, channels, head_size>::outs)& outs { head_rechannel.outs };

    void load_weights (std::vector<float>::iterator& weights)
    {
        std::vector<std::vector<float>> rechannel_weights (channels, std::vector<float> (in_size));
        for (int i = 0; i < channels; i++)
            for (int j = 0; j < in_size; j++)
                rechannel_weights[i][j] = *(weights++);
        rechannel.setWeights (rechannel_weights);

        RTNeural::modelt_detail::forEachInTuple ([&weights] (auto& layer, size_t)
                                                 { layer.load_weights (weights); },
                                                 layers);

        std::vector<std::vector<float>> head_rechannel_weights (head_size, std::vector<float> (channels));
        for (int i = 0; i < head_size; i++)
            for (int j = 0; j < channels; j++)
                head_rechannel_weights[i][j] = *(weights++);
        head_rechannel.setWeights (head_rechannel_weights);

        if (has_head_bias)
        {
            std::vector<float> head_rechannel_bias (head_size);
            for (int i = 0; i < head_size; i++)
                head_rechannel_bias[i] = *(weights++);
            head_rechannel.setBias (head_rechannel_bias.data());
        }
    }

    void forward (const Eigen::Matrix<T, in_size, 1>& ins,
                  const Eigen::Matrix<T, condition_size, 1>& condition,
                  Eigen::Matrix<T, channels, 1>& head_io)
    {
        rechannel.forward (ins);

        RTNeural::modelt_detail::forEachInTuple (
            [&] (auto& layer, auto index_t)
            {
                static constexpr size_t index = index_t;
                if constexpr (index == 0)
                    layer.forward (rechannel.outs, condition, head_io);
                else
                    layer.forward (std::get<index - 1> (layers).outs, condition, head_io);
            },
            layers);

        head_rechannel.forward (std::get<num_layers - 1> (layers).outs);
    }
};
} // namespace wavenet
