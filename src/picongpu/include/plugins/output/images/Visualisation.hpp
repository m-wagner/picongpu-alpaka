/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch, Felix Schmitt
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */



#pragma once

#include "simulation_defines.hpp"
#include "pmacc_types.hpp"


#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "fields/FieldJ.hpp"

#include "dimensions/DataSpace.hpp"
#include "dimensions/DataSpaceOperations.hpp"

#include "memory/buffers/GridBuffer.hpp"

#include "particles/memory/boxes/ParticlesBox.hpp"

#include "dataManagement/DataConnector.hpp"
#include "plugins/ILightweightPlugin.hpp"
#include "math/Vector.hpp"

#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/SharedBox.hpp"
#include "memory/boxes/PitchedBox.hpp"
#include "memory/buffers/GridBuffer.hpp"

#include "simulationControl/MovingWindow.hpp"

#include "mappings/kernel/MappingDescription.hpp"


//c includes
#include "sys/stat.h"
#include <cfloat>


#include "mappings/simulation/GridController.hpp"



#include <string>

#include "memory/boxes/PitchedBox.hpp"

#include "plugins/output/header/MessageHeader.hpp"
#include "plugins/output/GatherSlice.hpp"

#include "algorithms/GlobalReduce.hpp"
#include "memory/boxes/DataBoxDim1Access.hpp"
#include "nvidia/functors/Max.hpp"
#include "nvidia/atomic.hpp"
#include "mappings/elements/Vectorize.hpp"

namespace picongpu
{
using namespace PMacc;


// normalize EM fields to typical laser or plasma quantities
//-1: Auto:    enable adaptive scaling for each output
// 1: Laser:   typical fields calculated out of the laser amplitude
// 2: Drift:   outdated
// 3: PlWave:  typical fields calculated out of the plasma freq.,
//             assuming the wave moves approx. with c
// 4: Thermal: outdated
// 5: BlowOut: typical fields, assuming that a LWFA in the blowout
//             regime causes a bubble with radius of approx. the laser's
//             beam waist (use for bubble fields)
///  \return float3_X( tyBField, tyEField, tyCurrent )

template< int T >
struct typicalFields
{

    HDINLINE static float3_X get()
    {
        return float3_X(float_X(1.0), float_X(1.0), float_X(1.0));
    }
};

template< >
struct typicalFields < -1 >
{

    HDINLINE static float3_X get()
    {
        return float3_X(float_X(1.0), float_X(1.0), float_X(1.0));
    }
};

template< >
struct typicalFields < 1 >
{

    HDINLINE static float3_X get()
    {
#if !(EM_FIELD_SCALE_CHANNEL1 == 1 || EM_FIELD_SCALE_CHANNEL2 == 1 || EM_FIELD_SCALE_CHANNEL3 == 1)
        return float3_X(float_X(1.0), float_X(1.0), float_X(1.0));
#else
        const float_X tyCurrent = particles::TYPICAL_PARTICLES_PER_CELL * particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE
            * abs(BASE_CHARGE) / DELTA_T;
        const float_X tyEField = laserProfile::AMPLITUDE + FLT_MIN;
        const float_X tyBField = tyEField * MUE0_EPS0;

        return float3_X(tyBField, tyEField, tyCurrent);
#endif
    }
};


/* outdated drift normalization */
template< >
struct typicalFields < 2 >;

template< >
struct typicalFields < 3 >
{

    HDINLINE static float3_X get()
    {
#if !(EM_FIELD_SCALE_CHANNEL1 == 3 || EM_FIELD_SCALE_CHANNEL2 == 3 || EM_FIELD_SCALE_CHANNEL3 == 3)
        return float3_X(float_X(1.0), float_X(1.0), float_X(1.0));
#else
        const float_X lambda_pl = 2.0f * M_PI * SPEED_OF_LIGHT *
            sqrt(BASE_MASS * EPS0 / GAS_DENSITY / BASE_CHARGE / BASE_CHARGE);
        const float_X tyEField = lambda_pl * GAS_DENSITY / 3.0f / EPS0;
        const float_X tyBField = tyEField * MUE0_EPS0;
        const float_X tyCurrent = tyBField / MUE0;

        return float3_X(tyBField, tyEField, tyCurrent);
#endif
    }
};

/* outdated ELECTRON_TEMPERATURE normalization */
template< >
struct typicalFields < 4 >;

template< >
struct typicalFields < 5 >
{

    HDINLINE static float3_X get()
    {
#if !(EM_FIELD_SCALE_CHANNEL1 == 5 || EM_FIELD_SCALE_CHANNEL2 == 5 || EM_FIELD_SCALE_CHANNEL3 == 5)
        return float3_X(float_X(1.0), float_X(1.0), float_X(1.0));
#else
        const float_X tyEField = laserProfile::W0 * GAS_DENSITY / 3.0f / EPS0;
        const float_X tyBField = tyEField * MUE0_EPS0;
        const float_X tyCurrent = particles::TYPICAL_PARTICLES_PER_CELL * particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE
            * abs(BASE_CHARGE) / DELTA_T;

        return float3_X(tyBField, tyEField, tyCurrent);
#endif
    }
};

struct kernelPaintFields
{
template<class EBox, class BBox, class JBox, class Mapping, typename T_Acc>
DINLINE void operator()(
                                  const T_Acc& acc,
                                  EBox fieldE,
                                  BBox fieldB,
                                  JBox fieldJ,
                                  DataBox<PitchedBox<float3_X, DIM2> > image,
                                  DataSpace<DIM2> transpose,
                                  const int slice,
                                  const uint32_t globalOffset,
                                  const uint32_t sliceDim,
                                  Mapping mapper) const
{
    typedef typename MappingDesc::SuperCellSize Block;
    const DataSpace<simDim> elemSize(elemDim);
    const DataSpace<simDim> threadId( DataSpace<simDim>(threadIdx) * elemSize );
    const DataSpace<simDim> block = mapper.getSuperCellIndex(DataSpace<simDim > (blockIdx));
    const DataSpace<simDim> originCell(block * Block::toRT() + threadId);
    const DataSpace<simDim> blockOffset((block - mapper.getGuardingSuperCells()) * Block::toRT());

    namespace mapElem = mappings::elements;

    mapElem::vectorize<simDim>(
        [&]( const DataSpace<simDim>& idx )
        {
            const DataSpace<simDim> cell(originCell + idx);
            const DataSpace<simDim> realCell(cell - MappingDesc::SuperCellSize::toRT() * mapper.getGuardingSuperCells()); //delete guard from cell idx
            const DataSpace<DIM2> imageCell(
                                            realCell[transpose.x()],
                                            realCell[transpose.y()]);
            const DataSpace<simDim> realCell2(blockOffset + threadId + idx); //delete guard from cell idx

#if (SIMDIM==DIM3)
            uint32_t globalCell = realCell2[sliceDim] + globalOffset;

            if (globalCell != slice)
                return;
#endif
            // set fields of this cell to vars
            typename BBox::ValueType field_b = fieldB(cell);
            typename EBox::ValueType field_e = fieldE(cell);
            typename JBox::ValueType field_j = fieldJ(cell);

#if(SIMDIM==DIM3)
            field_j = float3_X(
                               field_j.x() * CELL_HEIGHT * CELL_DEPTH,
                               field_j.y() * CELL_WIDTH * CELL_DEPTH,
                               field_j.z() * CELL_WIDTH * CELL_HEIGHT
                               );
#elif (SIMDIM==DIM2)
            field_j = float3_X(
                               field_j.x() * CELL_HEIGHT,
                               field_j.y() * CELL_WIDTH,
                               field_j.z() * CELL_WIDTH * CELL_HEIGHT
                               );
#endif

            // reset picture to black
            //   color range for each RGB channel: [0.0, 1.0]
            float3_X pic = float3_X(0., 0., 0.);

            // typical values of the fields to normalize them to [0,1]
            //
            pic.x() = visPreview::preChannel1(field_b / typicalFields<EM_FIELD_SCALE_CHANNEL1>::get().x(),
                                              field_e / typicalFields<EM_FIELD_SCALE_CHANNEL1>::get().y(),
                                              field_j / typicalFields<EM_FIELD_SCALE_CHANNEL1>::get().z());
            pic.y() = visPreview::preChannel2(field_b / typicalFields<EM_FIELD_SCALE_CHANNEL2>::get().x(),
                                              field_e / typicalFields<EM_FIELD_SCALE_CHANNEL2>::get().y(),
                                              field_j / typicalFields<EM_FIELD_SCALE_CHANNEL2>::get().z());
            pic.z() = visPreview::preChannel3(field_b / typicalFields<EM_FIELD_SCALE_CHANNEL3>::get().x(),
                                              field_e / typicalFields<EM_FIELD_SCALE_CHANNEL3>::get().y(),
                                              field_j / typicalFields<EM_FIELD_SCALE_CHANNEL3>::get().z());
            //visPreview::preChannel1Col::addRGB(pic,
            //                                   visPreview::preChannel1(field_b * typicalFields<EM_FIELD_SCALE_CHANNEL1>::get().x(),
            //                                                           field_e * typicalFields<EM_FIELD_SCALE_CHANNEL1>::get().y(),
            //                                                           field_j * typicalFields<EM_FIELD_SCALE_CHANNEL1>::get().z()),
            //                                   visPreview::preChannel1_opacity);
            //visPreview::preChannel2Col::addRGB(pic,
            //                                   visPreview::preChannel2(field_b * typicalFields<EM_FIELD_SCALE_CHANNEL2>::get().x(),
            //                                                           field_e * typicalFields<EM_FIELD_SCALE_CHANNEL2>::get().y(),
            //                                                           field_j * typicalFields<EM_FIELD_SCALE_CHANNEL2>::get().z()),
            //                                   visPreview::preChannel2_opacity);
            //visPreview::preChannel3Col::addRGB(pic,
            //                                   visPreview::preChannel3(field_b * typicalFields<EM_FIELD_SCALE_CHANNEL3>::get().x(),
            //                                                           field_e * typicalFields<EM_FIELD_SCALE_CHANNEL3>::get().y(),
            //                                                           field_j * typicalFields<EM_FIELD_SCALE_CHANNEL3>::get().z()),
            //                                   visPreview::preChannel3_opacity);


            // draw to (perhaps smaller) image cell
            image(imageCell) = pic;
    },
    elemSize,
    mapElem::Contiguous()
    );
}
};

template< typename T_ElemSize = typename PMacc::math::CT::make_Int<simDim,1>::type::vector_type >
struct kernelPaintParticles3D
{
template<class ParBox, class Mapping, typename T_Acc>
DINLINE void
operator()(const T_Acc& acc,
                       ParBox pb,
                       DataBox<PitchedBox<float3_X, DIM2> > image,
                       DataSpace<DIM2> transpose,
                       int slice,
                       uint32_t globalOffset,
                       uint32_t sliceDim,
                       Mapping mapper) const
{
    typedef typename ParBox::FramePtr FramePtr;
    typedef typename MappingDesc::SuperCellSize Block;
    FramePtr frame;
    sharedMem(isValid, int);



    const DataSpace<simDim> threadId( DataSpace<simDim>(threadIdx) * DataSpace<simDim>(elemDim) );
    const DataSpace<DIM2> localCell(threadId[transpose.x()], threadId[transpose.y()]);
    const DataSpace<simDim> block = mapper.getSuperCellIndex(DataSpace<simDim > (blockIdx));
    const DataSpace<simDim> blockOffset((block - 1) * Block::toRT());

    namespace mapElem = mappings::elements;

    int g_localId = DataSpaceOperations<simDim>::template map<Block>( threadId );


    if (g_localId == 0)
        isValid = 0;
    __syncthreads();


        // counter is always DIM2
    typedef DataBox < PitchedBox< float_X, DIM2 > > SharedMem;
    sharedMemExtern(shBlock,float_X);

    const DataSpace<simDim> blockSize( DataSpace<simDim>(blockDim) * DataSpace<simDim>(elemDim));
    SharedMem counter(PitchedBox<float_X, DIM2 > ((float_X*) shBlock,
                                                  DataSpace<DIM2 > (),
                                                  blockSize[transpose.x()] * sizeof (float_X)));
    mapElem::vectorize<simDim>(
        [&]( const DataSpace<simDim>& idx )
        {

            //\todo: guard size should not be set to (fixed) 1 here
            const DataSpace<simDim> realCell(blockOffset + threadId + idx); //delete guard from cell idx
            bool isImageThread = false;
#if(SIMDIM==DIM3)
            uint32_t globalCell = realCell[sliceDim] + globalOffset;

            if (globalCell == slice)
#endif
            {
                nvidia::atomicAllExch(acc, &isValid, 1, ::alpaka::hierarchy::Threads()); /*WAW Error in cuda-memcheck racecheck*/
                isImageThread = true;
            }


            if (isImageThread)
            {
                const DataSpace<DIM2> t_idx(idx[transpose.x()], idx[transpose.y()]);
                counter(localCell + t_idx) = float_X(0.0);
            }
        },
        T_ElemSize::toRT(),
        mapElem::Contiguous()
    );
    __syncthreads();

    if (isValid == 0)
        return;

    frame = pb.getFirstFrame(block);

    while (frame.isValid()) //move over all Frames
    {
        mapElem::vectorize<simDim>(
            [&]( const DataSpace<simDim>& idx )
            {
                int localId = DataSpaceOperations<simDim>::template map<Block>( threadId + idx );

                PMACC_AUTO(particle, frame[localId]);
                if (particle[multiMask_] == 1)
                {
                    int cellIdx = particle[localCellIdx_];
                    // we only draw the first slice of cells in the super cell (z == 0)
                    const DataSpace<simDim> particleCellId(DataSpaceOperations<simDim>::template map<Block > (cellIdx));
#if(SIMDIM==DIM3)
                    uint32_t globalParticleCell = particleCellId[sliceDim] + globalOffset + blockOffset[sliceDim];
                    if (globalParticleCell == slice)
#endif
                    {
                        const DataSpace<DIM2> reducedCell(particleCellId[transpose.x()], particleCellId[transpose.y()]);
                        atomicAdd(&(counter(reducedCell)), particle[weighting_] / particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE,::alpaka::hierarchy::Threads());
                    }
                }
            },
            T_ElemSize::toRT(),
            mapElem::Contiguous()
        );

        frame = pb.getNextFrame(frame);
    }

    __syncthreads();

    mapElem::vectorize<simDim>(
        [&]( const DataSpace<simDim>& idx )
        {

            const DataSpace<simDim> realCell(blockOffset + threadId + idx); //delete guard from cell idx
                /*index in image*/
            DataSpace<DIM2> imageCell(
                                      realCell[transpose.x()],
                                      realCell[transpose.y()]);
            bool isImageThread = false;
#if(SIMDIM==DIM3)
            uint32_t globalCell = realCell[sliceDim] + globalOffset;

            if (globalCell == slice)
#endif
            {
                isImageThread = true;
            }
            if (isImageThread)
            {
                /** Note: normally, we would multiply by particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE again.
                 *  BUT: since we are interested in a simple value between 0 and 1,
                 *       we stay with this number (normalized to the order of macro
                 *       particles) and devide by the number of typical macro particles
                 *       per cell
                 */
                const DataSpace<DIM2> t_idx(idx[transpose.x()], idx[transpose.y()]);
                float_X value = counter(localCell + t_idx)
                    / float_X(particles::TYPICAL_PARTICLES_PER_CELL); // * particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE;
                if (value > 1.0) value = 1.0;

                //image(imageCell).x() = value;
                visPreview::preParticleDensCol::addRGB(image(imageCell),
                                                       value,
                                                       visPreview::preParticleDens_opacity);

                // cut to [0, 1]
                if (image(imageCell).x() < float_X(0.0)) image(imageCell).x() = float_X(0.0);
                if (image(imageCell).x() > float_X(1.0)) image(imageCell).x() = float_X(1.0);
                if (image(imageCell).y() < float_X(0.0)) image(imageCell).y() = float_X(0.0);
                if (image(imageCell).y() > float_X(1.0)) image(imageCell).y() = float_X(1.0);
                if (image(imageCell).z() < float_X(0.0)) image(imageCell).z() = float_X(0.0);
                if (image(imageCell).z() > float_X(1.0)) image(imageCell).z() = float_X(1.0);
            }
        },
        T_ElemSize::toRT(),
        mapElem::Contiguous()
    );
}
};

namespace vis_kernels
{

template< int T_elemSize = 1>
struct divideAnyCell
{
template<class Mem, typename Type, typename T_Acc>
DINLINE void operator()(const T_Acc& acc, Mem mem, uint32_t n, Type divisor) const
{
    namespace mapElem = mappings::elements;

    uint32_t stirdedTid = blockIdx.x * blockDim.x * elemDim.x + threadIdx.x;

    mapElem::vectorize<DIM1>(
        [&]( const int idx )
        {
            const int tid = stirdedTid + idx;

            if (tid >= n) return;

            const float3_X FLT3_MIN = float3_X(FLT_MIN, FLT_MIN, FLT_MIN);
            mem[tid] /= (divisor + FLT3_MIN);
        },
        T_elemSize
    );
}
};

template< int T_elemSize = 1>
struct channelsToRGB
{
template<class Mem, typename T_Acc>
DINLINE void operator()(const T_Acc& acc, Mem mem, uint32_t n) const
{
    namespace mapElem = mappings::elements;
    uint32_t stirdedTid = blockIdx.x * blockDim.x * elemDim.x + threadIdx.x;

    mapElem::vectorize<DIM1>(
        [&]( const int idx )
        {
            const int tid = stirdedTid + idx;
            if (tid >= n) return;

            float3_X rgb(float3_X::create(0.0));

            visPreview::preChannel1Col::addRGB(rgb,
                                               mem[tid].x(),
                                               visPreview::preChannel1_opacity);
            visPreview::preChannel2Col::addRGB(rgb,
                                               mem[tid].y(),
                                               visPreview::preChannel2_opacity);
            visPreview::preChannel3Col::addRGB(rgb,
                                               mem[tid].z(),
                                               visPreview::preChannel3_opacity);
            mem[tid] = rgb;
        },
        T_elemSize
    );
}
};

}

/**
 * Visualizes simulation data by writing png files.
 * Visulization is performed in an additional thread.
 */
template<class ParticlesType, class Output>
class Visualisation : public ILightweightPlugin
{
private:
    typedef MappingDesc::SuperCellSize SuperCellSize;


public:
    typedef typename ParticlesType::FrameType FrameType;
    typedef Output CreatorType;

    Visualisation(std::string name, Output output, uint32_t notifyPeriod, DataSpace<DIM2> transpose, float_X slicePoint) :
    m_output(output),
    analyzerName(name),
    cellDescription(NULL),
    particleTag(ParticlesType::FrameType::getName()),
    m_notifyPeriod(notifyPeriod),
    m_transpose(transpose),
    m_slicePoint(slicePoint),
    isMaster(false),
    header(NULL),
    reduce(1024),
    img(NULL)
    {
        sliceDim = 0;
        if (m_transpose.x() == 0 || m_transpose.y() == 0)
            sliceDim = 1;
        if ((m_transpose.x() == 1 || m_transpose.y() == 1) && sliceDim == 1)
            sliceDim = 2;

        Environment<>::get().PluginConnector().registerPlugin(this);
        Environment<>::get().PluginConnector().setNotificationPeriod(this, m_notifyPeriod);
    }

    virtual ~Visualisation()
    {
        /* wait that shared buffers can destroyed */
        m_output.join();
        if (m_notifyPeriod > 0)
        {
            __delete(img);
            MessageHeader::destroy(header);
        }
    }

    std::string pluginGetName() const
    {
        return "Visualisation";
    }

    void notify(uint32_t currentStep)
    {
        assert(cellDescription != NULL);
        const DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());
        Window window(MovingWindow::getInstance().getWindow(currentStep));

        /*sliceOffset is only used in 3D*/
        sliceOffset = (int) ((float_32) (window.globalDimensions.size[sliceDim]) * m_slicePoint) + window.globalDimensions.offset[sliceDim];

        if (!doDrawing())
        {
            return;
        }
        createImage(currentStep, window);
    }

    void setMappingDescription(MappingDesc *cellDescription)
    {
        assert(cellDescription != NULL);
        this->cellDescription = cellDescription;
    }

    void createImage(uint32_t currentStep, Window window)
    {
        DataConnector &dc = Environment<>::get().DataConnector();
        // Data does not need to be synchronized as visualization is
        // done at the device.
        FieldB *fieldB = &(dc.getData<FieldB > (FieldB::getName(), true));
        FieldE* fieldE = &(dc.getData<FieldE > (FieldE::getName(), true));
        FieldJ* fieldJ = &(dc.getData<FieldJ > (FieldJ::getName(), true));
        ParticlesType* particles = &(dc.getData<ParticlesType > (particleTag, true));

        /* wait that shared buffers can accessed without conflicts */
        m_output.join();

        uint32_t globalOffset = 0;
#if(SIMDIM==DIM3)
        globalOffset = Environment<simDim>::get().SubGrid().getLocalDomain().offset[sliceDim];
#endif

        typedef MappingDesc::SuperCellSize SuperCellSize;
        assert(cellDescription != NULL);
        //create image fields
        __picKernelArea_OPTI(kernelPaintFields)( *cellDescription, CORE + BORDER)
            (SuperCellSize::toRT().toDim3())
            (fieldE->getDeviceDataBox(),
             fieldB->getDeviceDataBox(),
             fieldJ->getDeviceDataBox(),
             img->getDeviceBuffer().getDataBox(),
             m_transpose,
             sliceOffset,
             globalOffset, sliceDim
             );

        // find maximum for img.x()/y and z and return it as float3_X
        int elements = img->getGridLayout().getDataSpace().productOfComponents();

        //Add one dimension access to 2d DataBox
        typedef DataBoxDim1Access<typename GridBuffer<float3_X, DIM2 >::DataBoxType> D1Box;
        D1Box d1access(img->getDeviceBuffer().getDataBox(), img->getGridLayout().getDataSpace());

        constexpr bool useElements = cupla::traits::IsThreadSeqAcc< cupla::AccThreadSeq >::value;

#if (EM_FIELD_SCALE_CHANNEL1 == -1 || EM_FIELD_SCALE_CHANNEL2 == -1 || EM_FIELD_SCALE_CHANNEL3 == -1)
        //reduce with functor max
        float3_X max = reduce(nvidia::functors::Max(),
                              d1access,
                              elements);
        //reduce with functor min
        //float3_X min = reduce(nvidia::functors::Min(),
        //                    d1access,
        //                    elements);
#if (EM_FIELD_SCALE_CHANNEL1 != -1 )
        max.x() = float_X(1.0);
#endif
#if (EM_FIELD_SCALE_CHANNEL2 != -1 )
        max.y() = float_X(1.0);
#endif
#if (EM_FIELD_SCALE_CHANNEL3 != -1 )
        max.z() = float_X(1.0);
#endif

        //We don't know the superCellSize at compile time
        // (because of the runtime dimension selection in any analyser),
        // thus we must use a one dimension kernel and no mapper
        if(useElements)
        {
            __cudaKernel_OPTI(vis_kernels::divideAnyCell<256>)(ceil((float_64) elements / 256), 256)(d1access, elements, max);
        }
        else
        {
            __cudaKernel(vis_kernels::divideAnyCell<>)(ceil((float_64) elements / 256), 256)(d1access, elements, max);
        }
#endif


        if(useElements)
        {
            // convert channels to RGB
            __cudaKernel_OPTI(vis_kernels::channelsToRGB<256>)(ceil((float_64) elements / 256), 256)(d1access, elements);
        }
        else
        {
            // convert channels to RGB
            __cudaKernel(vis_kernels::channelsToRGB<>)(ceil((float_64) elements / 256), 256)(d1access, elements);
        }

        // add density color channel
        DataSpace<simDim> blockSize(MappingDesc::SuperCellSize::toRT());
        DataSpace<DIM2> blockSize2D(blockSize[m_transpose.x()], blockSize[m_transpose.y()]);

        if(useElements)
        {
            //create image particles
            __picKernelArea_OPTI(kernelPaintParticles3D<SuperCellSize>)( *cellDescription, CORE + BORDER)
                (SuperCellSize::toRT().toDim3(), blockSize2D.productOfComponents() * sizeof (float_X))
                (particles->getDeviceParticlesBox(),
                 img->getDeviceBuffer().getDataBox(),
                 m_transpose,
                 sliceOffset,
                 globalOffset, sliceDim
                 );
        }
        else
        {
            //create image particles
            __picKernelArea(kernelPaintParticles3D<>)( *cellDescription, CORE + BORDER)
                (SuperCellSize::toRT().toDim3(),blockSize2D.productOfComponents() * sizeof (float_X))
                (particles->getDeviceParticlesBox(),
                 img->getDeviceBuffer().getDataBox(),
                 m_transpose,
                 sliceOffset,
                 globalOffset, sliceDim
                 );
        }

        // send the RGB image back to host
        img->deviceToHost();


        header->update(*cellDescription, window, m_transpose, currentStep);


        __getTransactionEvent().waitForFinished(); //wait for copy picture

        DataSpace<DIM2> size = img->getGridLayout().getDataSpace();

        PMACC_AUTO(hostBox, img->getHostBuffer().getDataBox());

        if (picongpu::white_box_per_GPU)
        {
            hostBox[0 ][0 ] = float3_X(1.0, 1.0, 1.0);
            hostBox[size.y() - 1 ][0 ] = float3_X(1.0, 1.0, 1.0);
            hostBox[0 ][size.x() - 1] = float3_X(1.0, 1.0, 1.0);
            hostBox[size.y() - 1 ][size.x() - 1] = float3_X(1.0, 1.0, 1.0);
        }
        PMACC_AUTO(resultBox, gather(hostBox, *header));
        if (isMaster)
        {
            m_output(resultBox.shift(header->window.offset), header->window.size, *header);
        }

    }

    void init()
    {
        if (m_notifyPeriod > 0)
        {
            assert(cellDescription != NULL);
            const DataSpace<simDim> localSize(cellDescription->getGridLayout().getDataSpaceWithoutGuarding());

            Window window(MovingWindow::getInstance().getWindow(0));
            sliceOffset = (int) ((float_32) (window.globalDimensions.size[sliceDim]) * m_slicePoint) + window.globalDimensions.offset[sliceDim];


            const DataSpace<simDim> gpus = Environment<simDim>::get().GridController().getGpuNodes();

            float_32 cellSizeArr[3] = {0, 0, 0};
            for (uint32_t i = 0; i < simDim; ++i)
                cellSizeArr[i] = cellSize[i];

            header = MessageHeader::create();
            header->update(*cellDescription, window, m_transpose, 0, cellSizeArr, gpus);

            bool isDrawing = doDrawing();
            isMaster = gather.init(isDrawing);
            reduce.participate(isDrawing);

            /* create memory for the local picture if the gpu participate on the visualization */
            if(isDrawing)
                img = new GridBuffer<float3_X, DIM2 > (header->node.maxSize);
        }
    }

    void pluginRegisterHelp(po::options_description& desc)
    {
        // nothing to do here
    }

private:

    bool doDrawing()
    {
        assert(cellDescription != NULL);
        const DataSpace<simDim> globalRootCellPos(Environment<simDim>::get().SubGrid().getLocalDomain().offset);
#if(SIMDIM==DIM3)
        const bool tmp = globalRootCellPos[sliceDim] + Environment<simDim>::get().SubGrid().getLocalDomain().size[sliceDim] > sliceOffset &&
            globalRootCellPos[sliceDim] <= sliceOffset;
        return tmp;
#else
        return true;
#endif
    }


    MappingDesc *cellDescription;
    SimulationDataId particleTag;

    GridBuffer<float3_X, DIM2 > *img;

    int sliceOffset;
    uint32_t m_notifyPeriod;
    float_X m_slicePoint;

    std::string analyzerName;


    DataSpace<DIM2> m_transpose;
    uint32_t sliceDim;

    MessageHeader* header;

    Output m_output;
    GatherSlice gather;
    bool isMaster;
    algorithms::GlobalReduce reduce;
};



}

