//=== logical_not.hpp -   Unary function ISNAN                 ------
//*-C++-*--/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines kernels for elementwise evaluation of ISNAN(x)
/// function that tests whether a tensor element is a NaN.
//===---------------------------------------------------------------------===//

#pragma once
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "vec_size_util.hpp"

#include "kernels/dpctl_tensor_types.hpp"

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace logical_not
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT, typename resT> struct LogicalNotFunctor
{
    static_assert(std::is_same_v<resT, bool>);

    using is_constant = typename std::false_type;
    // constexpr resT constant_value = resT{};
    using supports_vec = typename std::false_type;
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<tu_ns::is_complex<resT>, tu_ns::is_complex<argT>>>;

    resT operator()(const argT &in) const
    {
        using tu_ns::convert_impl;
        return !convert_impl<bool, argT>(in);
    }
};

template <typename argT,
          typename resT = bool,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using LogicalNotContigFunctor =
    elementwise_common::UnaryContigFunctor<argT,
                                           resT,
                                           LogicalNotFunctor<argT, resT>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using LogicalNotStridedFunctor =
    elementwise_common::UnaryStridedFunctor<argTy,
                                            resTy,
                                            IndexerT,
                                            LogicalNotFunctor<argTy, resTy>>;

template <typename argTy> struct LogicalNotOutputType
{
    using value_type = bool;
};

namespace hyperparam_detail
{

namespace vsu_ns = dpctl::tensor::kernels::vec_size_utils;

using vsu_ns::ContigHyperparameterSetDefault;
using vsu_ns::UnaryContigHyperparameterSetEntry;

template <typename argTy> struct LogicalNotContigHyperparameterSet
{
    using value_type =
        typename std::disjunction<ContigHyperparameterSetDefault<4u, 2u>>;

    constexpr static auto vec_sz = value_type::vec_sz;
    constexpr static auto n_vecs = value_type::n_vecs;
};

} // end of namespace hyperparam_detail

template <typename T1, typename T2, std::uint8_t vec_sz, std::uint8_t n_vecs>
class logical_not_contig_kernel;

template <typename argTy>
sycl::event
logical_not_contig_impl(sycl::queue &exec_q,
                        std::size_t nelems,
                        const char *arg_p,
                        char *res_p,
                        const std::vector<sycl::event> &depends = {})
{
    using LogicalNotHS =
        hyperparam_detail::LogicalNotContigHyperparameterSet<argTy>;
    static constexpr std::uint8_t vec_sz = LogicalNotHS::vec_sz;
    static constexpr std::uint8_t n_vecs = LogicalNotHS::n_vecs;

    return elementwise_common::unary_contig_impl<
        argTy, LogicalNotOutputType, LogicalNotContigFunctor,
        logical_not_contig_kernel, vec_sz, n_vecs>(exec_q, nelems, arg_p, res_p,
                                                   depends);
}

template <typename fnT, typename T> struct LogicalNotContigFactory
{
    fnT get()
    {
        fnT fn = logical_not_contig_impl<T>;
        return fn;
    }
};

template <typename fnT, typename T> struct LogicalNotTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::logical_not(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename LogicalNotOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3>
class logical_not_strided_kernel;

template <typename argTy>
sycl::event
logical_not_strided_impl(sycl::queue &exec_q,
                         std::size_t nelems,
                         int nd,
                         const ssize_t *shape_and_strides,
                         const char *arg_p,
                         ssize_t arg_offset,
                         char *res_p,
                         ssize_t res_offset,
                         const std::vector<sycl::event> &depends,
                         const std::vector<sycl::event> &additional_depends)
{
    return elementwise_common::unary_strided_impl<argTy, LogicalNotOutputType,
                                                  LogicalNotStridedFunctor,
                                                  logical_not_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct LogicalNotStridedFactory
{
    fnT get()
    {
        fnT fn = logical_not_strided_impl<T>;
        return fn;
    }
};

} // namespace logical_not
} // namespace kernels
} // namespace tensor
} // namespace dpctl
