/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once


#include "nvidia/functors/Assign.hpp"
#include "traits/GetValueType.hpp"
#include "types.h"

#include <boost/type_traits.hpp>

namespace PMacc
{
    namespace nvidia
    {
        namespace reduce
        {

            namespace kernel
            {

                template<
                    typename T_Val
                >
                struct Reduction
                {

                    static constexpr uint32_t kernelDim = DIM1;

                    template<
                        typename T_Acc,
                        typename Src,
                        typename Dest,
                        typename Functor,
                        typename Functor2>
                    DINLINE void operator()(
                        const T_Acc& acc,
                        const Src& src,
                        const uint32_t& src_count,
                        const Dest& dest,
                        const Functor& func,
                        const Functor2& func2) const
                    {

                        const DataSpace<dim> gridSize(::alpaka::workdiv::getWorkDiv<::alpaka::Grid, ::alpaka::Blocks>(acc));
                        const DataSpace<dim> blockSize(::alpaka::workdiv::getWorkDiv<::alpaka::Block, ::alpaka::Threads>(acc));
                        const DataSpace<dim> blockIndex(::alpaka::idx::getIdx<::alpaka::Grid, ::alpaka::Blocks>(acc));
                        const DataSpace<dim> threadIndex(::alpaka::idx::getIdx<::alpaka::Block, ::alpaka::Threads>(acc));

                        const uint32_t l_tid = threadIndex.x();
                        const uint32_t tid = blockIndex.x() * blockSize.x() + l_tid;
                        const uint32_t globalThreadCount = gridSize.x()* blockSize.x();

                        T_Val * const s_mem(acc.template getBlockSharedExternMem<T_Val>());

                        bool isActive = (tid < src_count);

                        /*wait that all shared memory is initialized*/
                        ::alpaka::block::sync::syncBlockThreads(acc);

                        /*fill shared mem*/
                        if (isActive)
                        {
                            T_Val r_value;

                            r_value = src[tid];
                            /*reduce not readed global memory to shared*/
                            uint32_t i = tid + globalThreadCount;
                            while (i < src_count)
                            {
                                func(r_value, src[i]);
                                i += globalThreadCount;
                            }
                            s_mem[l_tid] = r_value;
                        }

                        ::alpaka::block::sync::syncBlockThreads(acc);;
                        /*now reduce shared memory*/
                        uint32_t chunk_count = blockSize.x();
                        uint32_t active_threads;

                        while (chunk_count != 1)
                        {
                            const float half_threads = (float) chunk_count / 2.0f;
                            active_threads = static_cast<uint32_t>(::alpaka::math::trunc(acc, half_threads));
                            isActive = !(threadIndex.x() != 0 && l_tid >= active_threads);

                            chunk_count = ceilf(half_threads);
                            if (isActive)
                                func(s_mem[l_tid], s_mem[l_tid + chunk_count]);

                            ::alpaka::block::sync::syncBlockThreads(acc);;
                        }
                        if (isActive)
                            func2(dest[blockIndex.x()], s_mem[0]);
                    }
                };
            }
        }
    }
} //namespace PMacc

namespace alpaka
{
    namespace kernel
    {
        namespace traits
        {
            //#############################################################################
            //! The trait for getting the size of the block shared extern memory for a kernel.
            //#############################################################################
            template<
                typename T_Val,
                typename T_Acc>
            struct BlockSharedExternMemSizeBytes<
                ::PMacc::nvidia::reduce::kernel::Reduction<T_Val>,
                T_Acc>
            {
                //-----------------------------------------------------------------------------
                //! \return The size of the shared memory allocated for a block.
                //-----------------------------------------------------------------------------
                template<
                    typename TDim,
                    typename... TArgs>
                DINLINE static auto getBlockSharedExternMemSizeBytes(
                    const ::alpaka::Vec<TDim, ::alpaka::size::Size<T_Acc>>& blockThreadExtent,
                    const TArgs& ...)
                -> ::alpaka::size::Size<T_Acc>
                {
                    return blockThreadExtent.prod() * sizeof(T_Val);
                }
            };
        }
    }
}

namespace PMacc
{
    namespace nvidia
    {
        namespace reduce
        {
            class Reduce
            {
            public:

                /* Constructor
                 * Don't create a instance before you have set you cuda device!
                 * @param byte how many bytes in global gpu memory can reserved for the reduce algorithm
                 * @param sharedMemByte limit the usage of shared memory per block on gpu
                 */
                HINLINE Reduce(const uint32_t byte, const uint32_t sharedMemByte = 4 * 1024) :
                byte(byte), sharedMemByte(sharedMemByte), reduceBuffer(NULL)
                {

                    reduceBuffer = new GridBuffer<char, DIM1 > (DataSpace<DIM1 > (byte));
                }

                /* Reduce elements in global gpu memory
                 *
                 * @param func binary functor for reduce which takes two arguments, first argument is the source and get the new reduced value.
                 * Functor must specialize the function getMPI_Op.
                 * @param src a class or a pointer where the reduce algorithm can access the value by operator [] (one dimension access)
                 * @param n number of elements to reduce
                 *
                 * @return reduced value
                 */
                template<class Functor, typename Src>
                HINLINE typename traits::GetValueType<Src>::ValueType operator()(Functor func, Src src, uint32_t n)
                {
                   /* - the result of a functor can be a reference or a const value
                    * - it is not allowed to create const or reference memory
                    *   thus we remove `references` and `const` qualifiers */
                   typedef typename boost::remove_const<
                               typename boost::remove_reference<
                                   typename traits::GetValueType<Src>::ValueType
                               >::type
                           >::type Type;

                    uint32_t blockcount = optimalThreadsPerBlock(n, sizeof (Type));

                    uint32_t n_buffer = byte / sizeof (Type);

                    uint32_t threads = n_buffer * blockcount * 2; /* x2 is used thus we can use all byte in Buffer, after we calculate threads/2 */



                    if (threads > n) threads = n;
                    Type* dest = (Type*) reduceBuffer->getDeviceBuffer().getBasePointer();

                    uint32_t blocks = threads / 2 / blockcount;
                    if (blocks == 0) blocks = 1;
                    __cudaKernel(kernel::Reduction < Type >)
                        (blocks, blockcount)
                        (src, n, dest, func, PMacc::nvidia::functors::Assign());
                    n = blocks;
                    blockcount = optimalThreadsPerBlock(n, sizeof (Type));
                    blocks = n / 2 / blockcount;
                    if (blocks == 0 && n > 1) blocks = 1;


                    while (blocks != 0)
                    {
                        if (blocks > 1)
                        {
                            uint32_t blockOffset = ceil((double) blocks / blockcount);
                            uint32_t useBlocks = blocks - blockOffset;
                            uint32_t problemSize = n - (blockOffset * blockcount);
                            Type* srcPtr = dest + (blockOffset * blockcount);

                            __cudaKernel(kernel::Reduction < Type >)(useBlocks, blockcount)(srcPtr, problemSize, dest, func, func);
                            blocks = blockOffset*blockcount;
                        }
                        else
                        {

                            __cudaKernel(kernel::Reduction < Type >)(blocks, blockcount)(dest, n, dest, func,
                                                                                         PMacc::nvidia::functors::Assign());
                        }

                        n = blocks;
                        blockcount = optimalThreadsPerBlock(n, sizeof (Type));
                        blocks = n / 2 / blockcount;
                        if (blocks == 0 && n > 1) blocks = 1;
                    }

                    reduceBuffer->deviceToHost();
                    __getTransactionEvent().waitForFinished();
                    return *((Type*) (reduceBuffer->getHostBuffer().getBasePointer()));

                }

                virtual ~Reduce()
                {
                    __delete(reduceBuffer);
                }

            private:

                /* calculate number of threads per block
                 * @param threads maximal number of threads per block
                 * @return number of threads per block
                 */
                HINLINE uint32_t getThreadsPerBlock(uint32_t threads)
                {
                    /// \todo this list is not complete
                    ///        extend it and maybe check for sm_version
                    ///        and add possible threads accordingly.
                    ///        maybe this function should be exported
                    ///        to a more general nvidia class, too.
                    if (threads >= 512) return 512;
                    if (threads >= 256) return 256;
                    if (threads >= 128) return 128;
                    if (threads >= 64) return 64;
                    if (threads >= 32) return 32;
                    if (threads >= 16) return 16;
                    if (threads >= 8) return 8;
                    if (threads >= 4) return 4;
                    if (threads >= 2) return 2;

                    return 1;
                }

                /*calculate optimal number of threads per block with respect to shared memory limitations
                 * @param n number of elements to reduce
                 * @param sizePerElement size in bytes per elements
                 * @return optimal count of threads per block to solve the problem
                 */
                HINLINE uint32_t optimalThreadsPerBlock(uint32_t n, uint32_t sizePerElement)
                {
                    uint32_t const sharedBorder = sharedMemByte / sizePerElement;
                    return getThreadsPerBlock(std::min(sharedBorder, n));
                }

                /*global gpu buffer for reduce steps*/
                GridBuffer<char, DIM1 > *reduceBuffer;
                /*buffer size limit in bytes on gpu*/
                uint32_t byte;
                /*shared memory limit in byte for one block*/
                uint32_t sharedMemByte;

            };
        }
    }
}
