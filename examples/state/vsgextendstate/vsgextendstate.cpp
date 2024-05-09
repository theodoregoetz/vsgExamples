#include <string>
#include <iostream>

#include "vsg/all.h"


std::string geomVertShaderSource{R"(
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(push_constant) uniform PushConstants { mat4 projection; mat4 modelView; };
layout(location = 0) in vec3 vertex;
out gl_PerVertex { vec4 gl_Position; };
void main() { gl_Position = (projection * modelView) * vec4(vertex, 1.0); }
)"};

std::string resolveVertShaderSource{R"(
#version 450

/*
 * Copyright (c) 2023, Google
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	 http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#extension GL_ARB_separate_shader_objects : enable

void main()
{
	const vec2 fullscreen_triangle[] =
	{
		vec2(-1.0f,  3.0f),
		vec2(-1.0f, -1.0f),
		vec2( 3.0f, -1.0f),
	};
	const vec2 vertex = fullscreen_triangle[gl_VertexIndex % 3];
	gl_Position = vec4(vertex, 0.0f, 1.0f);
}
)"};

std::string geomFragShaderSource{R"(
#version 450

/*
 * Copyright (c) 2023, Google
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	 http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#extension GL_ARB_separate_shader_objects : enable
#pragma import_defines ( OIT )

#define OIT 1
#if defined(OIT)
    layout(set = 0, binding = 0) uniform OITConstants
    {
        uint  fragmentMaxCount;
        uint  sortedFragmentCount;
    } oitConstants;

    layout(set = 0, binding = 1, r32ui) uniform uimage2D linkedListHeadTex;

    layout(set = 0, binding = 2) buffer FragmentBuffer
    {
        uvec3 data[];
    } fragmentBuffer;

    layout(set = 0, binding = 3) buffer FragmentCounter
    {
        uint value;
    } fragmentCounter;
#else
    layout(location = 0) out vec4 outFragColor;
#endif

void main()
{
    float a = gl_PrimitiveID*0.1;
    float b = 1.0-gl_PrimitiveID*0.1;
    vec4 color;
    if (gl_PrimitiveID % 2 == 0)
        color = vec4(a, b, 0, 0.5);
    else
        color = vec4(b, a, 0, 0.5);

    #if defined(OIT)
        // Get the next fragment index
        const uint nextFragmentIndex = atomicAdd(fragmentCounter.value, 1U);

        // Ignore the fragment if the fragment buffer is full
        if(nextFragmentIndex >= oitConstants.fragmentMaxCount)
        {
            discard;
        }

        // Update the linked list head
        const uint previousFragmentIndex = imageAtomicExchange(linkedListHeadTex, ivec2(gl_FragCoord.xy), nextFragmentIndex);

        // Add the fragment to the buffer
        fragmentBuffer.data[nextFragmentIndex] = uvec3(previousFragmentIndex, packUnorm4x8(color), floatBitsToUint(gl_FragCoord.z));
    #else
        outFragColor = color;
    #endif
}
)"};

std::string resolveFragShaderSource{R"(
#version 450

/*
 * Copyright (c) 2023, Google
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	 http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#extension GL_ARB_separate_shader_objects : enable

#define LINKED_LIST_END_SENTINEL	0xFFFFFFFFU

// For performance reasons, this should be kept as low as result correctness allows.
#define SORTED_FRAGMENT_MAX_COUNT	16U

layout(set = 0, binding = 0) uniform OITConstants
{
	uint  fragmentMaxCount;
	uint  sortedFragmentCount;
} oitConstants;

layout(set = 0, binding = 1, r32ui) uniform uimage2D linkedListHeadTex;

layout(set = 0, binding = 2) buffer FragmentBuffer
{
	uvec3 data[];
} fragmentBuffer;

layout(set = 0, binding = 3) buffer FragmentCounter
{
	uint value;
} fragmentCounter;

layout (location = 0) out vec4 outFragColor;

// Blend two colors.
// The alpha channel keeps track of the amount of visibility of the background.
vec4 blendColors(uint packedSrcColor, vec4 dstColor)
{
	const vec4 srcColor = unpackUnorm4x8(packedSrcColor);
	return vec4(
		mix(dstColor.rgb, srcColor.rgb, srcColor.a),
		dstColor.a * (1.0f - srcColor.a));
}

// Sort and blend fragments from the linked list.
// For performance reasons, the maximum number of sorted fragments is limited.
// Approximations are used when the number of fragments is over the limit.
vec4 mergeSort(uint firstFragmentIndex)
{
	// Fragments are sorted from back to front.
	// e.g. sortedFragments[0] is the farthest from the camera.
	uvec2 sortedFragments[SORTED_FRAGMENT_MAX_COUNT];
	uint sortedFragmentCount = 0U;
	const uint kSortedFragmentMaxCount = min(oitConstants.sortedFragmentCount, SORTED_FRAGMENT_MAX_COUNT);

	vec4 color = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    bool initialColorSet = false;
	uint fragmentIndex = firstFragmentIndex;
	while(fragmentIndex != LINKED_LIST_END_SENTINEL)
	{
		const uvec3 fragment = fragmentBuffer.data[fragmentIndex];
		fragmentIndex = fragment.x;

		if(sortedFragmentCount < kSortedFragmentMaxCount)
		{
			// There is still room in the sorted list.
			// Insert the fragment so that the list stay sorted.
			uint i = sortedFragmentCount;
			for(; (i > 0) && (fragment.z < sortedFragments[i - 1].y); --i)
			{
				sortedFragments[i] = sortedFragments[i - 1];
			}
			sortedFragments[i] = fragment.yz;
			++sortedFragmentCount;
		}
		else if(sortedFragments[0].y < fragment.z)
		{
			// The fragment is closer than the farthest sorted one.
			// First, make room by blending the farthest fragment from the sorted list.
			// Then, insert the fragment in the sorted list.
            // This is an approximation.
            if (initialColorSet) {
                color = blendColors(sortedFragments[0].x, color);
            } else {
                color = unpackUnorm4x8(sortedFragments[0].x);
                initialColorSet = true;
            }
			uint i = 0;
			for(; (i < kSortedFragmentMaxCount - 1) && (sortedFragments[i + 1].y < fragment.z); ++i)
			{
				sortedFragments[i] = sortedFragments[i + 1];
			}
			sortedFragments[i] = fragment.yz;
		}
		else
		{
			// The next fragment is farther than any of the sorted ones.
			// Blend it early.
            // This is an approximation.
            if (initialColorSet) {
                color = blendColors(fragment.y, color);
            } else {
                color = unpackUnorm4x8(fragment.y);
                initialColorSet = true;
            }
		}
	}

	// Early return if there are no fragments.
	if(sortedFragmentCount == 0)
	{
		return vec4(0.0f);
	}

	// Blend the sorted fragments to get the final color.
	for(int i = 0; i < sortedFragmentCount; ++i)
	{
        if (i == 0 && !initialColorSet) {
            color = unpackUnorm4x8(sortedFragments[i].x);
        } else {
    		color = blendColors(sortedFragments[i].x, color);
        }
	}
	color.a = 1.0f - color.a;
	return color;
}

void main()
{
	// Reset the atomic counter for the next frame.
	// Note that we don't care about atomicity here, as all threads will write the same value.
	fragmentCounter.value = 0;

	// Get the first fragment index in the linked list.
	uint fragmentIndex = imageLoad(linkedListHeadTex, ivec2(gl_FragCoord.xy)).r;
	// Reset the list head for the next frame.
	imageStore(linkedListHeadTex, ivec2(gl_FragCoord.xy), uvec4(LINKED_LIST_END_SENTINEL, 0, 0, 0));

	// Compute the final color.
	outFragColor = mergeSort(fragmentIndex);
}
)"};

int main(int argc, char** argv)
{
    try
    {
        auto windowTraits = vsg::WindowTraits::create();
        /// set up defaults and read command line arguments to override them
        vsg::CommandLine arguments(&argc, argv);
        windowTraits->debugLayer = arguments.read({"--debug", "-d"});
        windowTraits->apiDumpLayer = arguments.read({"--api", "-a"});
        windowTraits->windowTitle = "vsgextendstate";
        windowTraits->deviceFeatures = vsg::DeviceFeatures::create();
        windowTraits->deviceFeatures->get().geometryShader = VK_TRUE; // for gl_PrimitiveID
        windowTraits->deviceFeatures->get().fragmentStoresAndAtomics = VK_TRUE;
        windowTraits->vulkanVersion = VK_API_VERSION_1_1;

        // create window now we have configured the windowTraits to set up the required features
        auto window = vsg::Window::create(windowTraits);
        if (!window)
        {
            std::cout << "Unable to create window" << std::endl;
            return 1;
        }

        int gpuNumber{0};
        auto instance = window->getOrCreateInstance();
        (void)window->getOrCreateSurface(); // fulfill contract of getOrCreatePhysicalDevice();
        auto& physicalDevices = instance->getPhysicalDevices();
        if (physicalDevices.empty()) {
            throw std::out_of_range{"No physical GPU devices reported."};
        } else if (gpuNumber < 0 || gpuNumber >= static_cast<int32_t>(physicalDevices.size())) {
            throw std::out_of_range{
                std::string{"Invalid GPU number: "}
                    .append(std::to_string(gpuNumber))
                    .append(". ")
                    .append("You may either specify -1 to let the system choose ")
                    .append("or specify a physical GPU device between 0 and ")
                    .append(std::to_string(physicalDevices.size() - 1u))
                    .append(" inclusive.")};
        }
        window->setPhysicalDevice(physicalDevices[gpuNumber]);

        auto lookAt = vsg::LookAt::create(
            vsg::dvec3{3, 2, 2},
            vsg::dvec3{0.2, 0.2, 0.2},
            vsg::dvec3{0, 0, 1}
        );
        auto perspective = vsg::Perspective::create();
        auto viewportState = vsg::ViewportState::create(window->extent2D());
        auto camera = vsg::Camera::create(perspective, lookAt, viewportState);
        auto viewer = vsg::Viewer::create();

        viewer->addWindow(window);
        viewer->addEventHandler(vsg::CloseHandler::create(viewer));
        viewer->addEventHandler(vsg::Trackball::create(camera));

        // create the geometry frame buffer
        vsg::SubpassDescription subpassDescription{};
        subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

        // geometry render pass doesn't need any output attachments
        auto geomRenderPass{vsg::RenderPass::create(window->getOrCreateDevice(), vsg::RenderPass::Attachments{},
            vsg::RenderPass::Subpasses{{subpassDescription}}, vsg::RenderPass::Dependencies{})};
        auto geomFramebuffer{vsg::Framebuffer::create(geomRenderPass, vsg::ImageViews{},
            window->extent2D().width, window->extent2D().height, 1/*layers*/)};

        // create a SSBO buffer for maintaining the linked list current node count
        auto linkedListCurSizeBuffer = vsg::createBufferAndMemory(
            window->getOrCreateDevice(),
            sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        auto linkedListCurSizeInfo{vsg::BufferInfo::create()};
        linkedListCurSizeInfo->buffer = linkedListCurSizeBuffer;
        linkedListCurSizeInfo->offset = 0;
        linkedListCurSizeInfo->range = sizeof(uint32_t);

        // create a uniform buffer for specifying the oit algorithm constants
        constexpr uint32_t kSortedFragmentMaxCount = 16;
        constexpr uint32_t kFragmentsPerPixelAverage = kSortedFragmentMaxCount/2;
        struct OITConstants
        {
            uint32_t fragmentMaxCount;
            uint32_t sortedFragmentCount;
        };
        using OITConstantsValue = vsg::Value<OITConstants>;
        auto oitConstants{OITConstantsValue::create()};
        oitConstants->value().fragmentMaxCount =
            window->extent2D().width * window->extent2D().height * kFragmentsPerPixelAverage;
        oitConstants->value().sortedFragmentCount = kSortedFragmentMaxCount;

        // create a texture for the headIndex to track the head index of each fragment
        auto headIndexImage{vsg::Image::create()};
        headIndexImage->format = VK_FORMAT_R32_UINT;
        headIndexImage->extent.width = window->extent2D().width;
        headIndexImage->extent.height = window->extent2D().height;
        headIndexImage->extent.depth = 1;
        headIndexImage->mipLevels = 1;
        headIndexImage->arrayLayers = 1;
        headIndexImage->samples = VK_SAMPLE_COUNT_1_BIT;
        #if defined VK_USE_PLATFORM_MACOS_MVK
            // SRS - On macOS/iOS use linear tiling for atomic image access.
            // see https://github.com/KhronosGroup/MoltenVK/issues/1027
            headIndexImage->tiling = VK_IMAGE_TILING_LINEAR;
        #else
            headIndexImage->tiling = VK_IMAGE_TILING_OPTIMAL;
        #endif
        headIndexImage->usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

        auto headIndexImageInfo{vsg::ImageInfo::create()};
        headIndexImageInfo->imageView = vsg::createImageView(
            window->getOrCreateDevice(), headIndexImage, VK_IMAGE_ASPECT_COLOR_BIT);
        headIndexImageInfo->imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // create a buffer for the fragment linked list
        auto linkedListBufferSize{static_cast<VkDeviceSize>(
            sizeof(vsg::uivec3) * oitConstants->value().fragmentMaxCount)};
        auto linkedListBuffer{vsg::createBufferAndMemory(
            window->getOrCreateDevice(), linkedListBufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        auto linkedListInfo{vsg::BufferInfo::create()};
        linkedListInfo->buffer = linkedListBuffer;
        linkedListInfo->offset = 0;
        linkedListInfo->range = linkedListBufferSize;

        // initialize shader buffer resources
        {
            // create a command to clear the linked list current size to zero
            class FillBuffer : public vsg::Inherit<vsg::Command, FillBuffer>
            {
            public:
                vsg::ref_ptr<vsg::Buffer> buffer;
                VkDeviceSize dstOffset;
                VkDeviceSize dataSize;
                uint32_t data;

                FillBuffer(vsg::ref_ptr<vsg::Buffer> in_buffer,
                           VkDeviceSize in_dstOffset,
                           VkDeviceSize in_dataSize,
                           uint32_t in_data)
                    : Inherit{}
                    , buffer{in_buffer}
                    , dstOffset{in_dstOffset}
                    , dataSize{in_dataSize}
                    , data{in_data}
                {
                }

                FillBuffer(FillBuffer const& rhs, vsg::CopyOp const& copyop = {})
                    : Inherit(rhs, copyop)
                {
                }

                void record(vsg::CommandBuffer& commandBuffer) const override
                {
                    vkCmdFillBuffer(commandBuffer, buffer->vk(commandBuffer.deviceID), dstOffset, dataSize, data);
                }

            protected:
                ~FillBuffer() override = default;
            };
            auto linkedListCurSizeClearCommand{FillBuffer::create(linkedListCurSizeBuffer, 0, sizeof(uint32_t), 0)};

            // create a pipeline barrier command to change the headIndexImage's layout from UNDEFINED to GENERAL
            auto headIndexImageBarrier{vsg::ImageMemoryBarrier::create()};
            headIndexImageBarrier->srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            headIndexImageBarrier->dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            headIndexImageBarrier->oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            headIndexImageBarrier->newLayout = VK_IMAGE_LAYOUT_GENERAL;
            headIndexImageBarrier->image = headIndexImage;
            headIndexImageBarrier->subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            headIndexImageBarrier->subresourceRange.baseArrayLayer = 0;
            headIndexImageBarrier->subresourceRange.layerCount = 1;
            headIndexImageBarrier->subresourceRange.levelCount = 1;

            auto headIndexImageBarrierCommand{vsg::PipelineBarrier::create(
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, headIndexImageBarrier)};

            // create a clear command to reset the headIndexImage texture with end-of-list values, 0xffffffff
            VkClearColorValue headImageClearColor;
            for (auto& uint32 : headImageClearColor.uint32)
                uint32 = 0xffffffff;

            VkImageSubresourceRange subresRange{};
            subresRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            subresRange.levelCount = 1;
            subresRange.layerCount = 1;

            auto headIndexImageClearColorCommand{vsg::ClearColorImage::create()};
            headIndexImageClearColorCommand->image = headIndexImage;
            headIndexImageClearColorCommand->imageLayout = headIndexImageInfo->imageLayout;
            headIndexImageClearColorCommand->color = headImageClearColor;
            headIndexImageClearColorCommand->ranges.push_back(subresRange);

            // execute the command to perform the image layout transition and initialization
            auto queueFamilyIndex{window->getOrCreatePhysicalDevice()->getQueueFamily(VK_QUEUE_GRAPHICS_BIT)};
            auto commandPool{vsg::CommandPool::create(window->getOrCreateDevice(), queueFamilyIndex)};
            auto queue{window->getOrCreateDevice()->getQueue(queueFamilyIndex)};
            vsg::submitCommandsToQueue(commandPool, nullptr, 0, queue,
                [&](vsg::CommandBuffer& commandBuffer){
                    linkedListCurSizeClearCommand->record(commandBuffer);
                    headIndexImageBarrierCommand->record(commandBuffer);
                    headIndexImageClearColorCommand->record(commandBuffer);
                });
        }

        // geometry pipeline construction
        vsg::DescriptorSetLayoutBindings geomDescriptorBindings{
            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
        };
        auto geomOITConstantsDesc{vsg::DescriptorBuffer::create(oitConstants,
            0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)};
        auto geomHeadIndexImageDesc{vsg::DescriptorImage::create(headIndexImageInfo,
            1, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)};
        auto geomLinkedListDesc{vsg::DescriptorBuffer::create(vsg::BufferInfoList{linkedListInfo},
            2, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)};
        auto geomLinkedListCurSizeInfoDesc{vsg::DescriptorBuffer::create(vsg::BufferInfoList{linkedListCurSizeInfo},
            3, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)};

        auto geomDescriptorSetLayout{vsg::DescriptorSetLayout::create(geomDescriptorBindings)};
        auto geomDescriptorSet{vsg::DescriptorSet::create(geomDescriptorSetLayout,
            vsg::Descriptors{geomOITConstantsDesc,geomHeadIndexImageDesc,geomLinkedListDesc,geomLinkedListCurSizeInfoDesc})};

        vsg::VertexInputState::Bindings geomVertexBindingsDescriptions{
            VkVertexInputBindingDescription{0, sizeof(vsg::vec3), VK_VERTEX_INPUT_RATE_VERTEX}, // vertex data
        };

        vsg::VertexInputState::Attributes geomVertexAttributeDescriptions{
            VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, // vertex data
        };

        auto inputAssemblyState{vsg::InputAssemblyState::create(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)};
        auto rasterizationState{vsg::RasterizationState::create()};
        rasterizationState->cullMode = VK_CULL_MODE_NONE;
        auto depthStencilState{vsg::DepthStencilState::create()};
        depthStencilState->depthTestEnable = VK_FALSE;
        depthStencilState->depthWriteEnable = VK_FALSE;
        depthStencilState->back.compareOp = VK_COMPARE_OP_ALWAYS;
        vsg::GraphicsPipelineStates geomPipelineStates{
            vsg::VertexInputState::create(geomVertexBindingsDescriptions, geomVertexAttributeDescriptions),
            inputAssemblyState,
            rasterizationState,
            vsg::MultisampleState::create(),
            vsg::ColorBlendState::create(),
            depthStencilState
        };

        vsg::PushConstantRanges pushConstantRanges{
            {VK_SHADER_STAGE_VERTEX_BIT, 0, 128} // projection, view, and model matrices, actual push constant calls automatically provided by the VSG's RecordTraversal
        };
        auto geomPipelineLayout{vsg::PipelineLayout::create(
            vsg::DescriptorSetLayouts{geomDescriptorSetLayout}, pushConstantRanges)};

        auto geomVertShader = vsg::ShaderStage::create(VK_SHADER_STAGE_VERTEX_BIT, "main", geomVertShaderSource);
        auto geomFragShader = vsg::ShaderStage::create(VK_SHADER_STAGE_FRAGMENT_BIT, "main", geomFragShaderSource);
        auto geomShaderStages{vsg::ShaderStages{geomVertShader, geomFragShader}};
        auto geomGraphicsPipeline{vsg::GraphicsPipeline::create(
            geomPipelineLayout, geomShaderStages, geomPipelineStates)};

        auto geomBindGraphicsPipeline{vsg::BindGraphicsPipeline::create(geomGraphicsPipeline)};
        auto geomBindDescriptorSet{vsg::BindDescriptorSet::create(
            VK_PIPELINE_BIND_POINT_GRAPHICS, geomGraphicsPipeline->layout, 0, geomDescriptorSet)};

        // resolve pipeline construction
        vsg::DescriptorSetLayoutBindings resolveDescriptorBindings{
            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
        };
        auto resolveOITConstantsDesc{vsg::DescriptorBuffer::create(oitConstants,
            0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)};
        auto resolveHeadIndexImageDesc{vsg::DescriptorImage::create(headIndexImageInfo,
            1, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)};
        auto resolveLinkedListDesc{vsg::DescriptorBuffer::create(vsg::BufferInfoList{linkedListInfo},
            2, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)};
        auto resolveLinkedListCurSizeInfoDesc{vsg::DescriptorBuffer::create(vsg::BufferInfoList{linkedListCurSizeInfo},
            3, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)};

        auto resolveDescriptorSetLayout{vsg::DescriptorSetLayout::create(resolveDescriptorBindings)};
        auto resolveDescriptorSet{vsg::DescriptorSet::create(resolveDescriptorSetLayout,
            vsg::Descriptors{resolveOITConstantsDesc,resolveHeadIndexImageDesc,resolveLinkedListDesc,resolveLinkedListCurSizeInfoDesc})};

        VkPipelineColorBlendAttachmentState resolveColorBlendState{};
        resolveColorBlendState.blendEnable         = VK_TRUE;
        resolveColorBlendState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        resolveColorBlendState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        resolveColorBlendState.colorBlendOp        = VK_BLEND_OP_ADD;
        resolveColorBlendState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        resolveColorBlendState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        resolveColorBlendState.alphaBlendOp        = VK_BLEND_OP_ADD;
        resolveColorBlendState.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        vsg::GraphicsPipelineStates resolvePipelineStates{
            vsg::VertexInputState::create(),
            inputAssemblyState,
            rasterizationState,
            vsg::MultisampleState::create(),
            vsg::ColorBlendState::create(vsg::ColorBlendState::ColorBlendAttachments{{resolveColorBlendState}}),
            depthStencilState
        };
        auto resolvePipelineLayout{vsg::PipelineLayout::create(
            vsg::DescriptorSetLayouts{resolveDescriptorSetLayout}, vsg::PushConstantRanges{})};

        auto resolveVertShader = vsg::ShaderStage::create(VK_SHADER_STAGE_VERTEX_BIT, "main", resolveVertShaderSource);
        auto resolveFragShader = vsg::ShaderStage::create(VK_SHADER_STAGE_FRAGMENT_BIT, "main", resolveFragShaderSource);
        auto resolveShaderStages{vsg::ShaderStages{resolveVertShader, resolveFragShader}};
        auto resolveGraphicsPipeline{vsg::GraphicsPipeline::create(
            resolvePipelineLayout, resolveShaderStages, resolvePipelineStates)};

        auto resolveBindGraphicsPipeline{vsg::BindGraphicsPipeline::create(resolveGraphicsPipeline)};
        auto resolveBindDescriptorSet{vsg::BindDescriptorSet::create(
            VK_PIPELINE_BIND_POINT_GRAPHICS, resolveGraphicsPipeline->layout, 0, resolveDescriptorSet)};

        // setup the geometry state group
        auto geomStateGroup{vsg::StateGroup::create()};
        geomStateGroup->add(geomBindGraphicsPipeline);
        geomStateGroup->add(geomBindDescriptorSet);

        // render the geometry scene
        std::vector<vsg::ref_ptr<vsg::vec3Array>> vertices{{
            vsg::vec3Array::create({
                {0, 0, 0},
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1},
                {1, 0, 1},
                {1, 1, 1}
            }),
            vsg::vec3Array::create({
                {0, 0, 0.5},
                {1, 0, 0.5},
                {0, 1, 0.5},
                {0, 0, 1.5},
                {1, 0, 1.5},
                {1, 1, 1.5}
            })
        }};

        for (auto& verts : vertices) {
            vsg::DataList vertexArrays{{verts}};
            auto vertexDraw{vsg::VertexDraw::create()};
            vertexDraw->assignArrays(vertexArrays);
            vertexDraw->vertexCount = verts->width();
            vertexDraw->instanceCount = 1;
            geomStateGroup->addChild(vertexDraw);
        }

        // setup the resolve state group
        auto resolveStateGroup{vsg::StateGroup::create()};
        resolveStateGroup->add(resolveBindGraphicsPipeline);
        resolveStateGroup->add(resolveBindDescriptorSet);

        // render the resolve scene which is just a triangle fabricated in the resolve vertex shader
        class FabricatedTriangleDraw : public vsg::Inherit<vsg::Command, FabricatedTriangleDraw>
        {
        public:
            FabricatedTriangleDraw() = default;
            FabricatedTriangleDraw(FabricatedTriangleDraw const& rhs, vsg::CopyOp const& copyop = {})
                : Inherit(rhs, copyop)
            {
            }

            void record(vsg::CommandBuffer& commandBuffer) const override
            {
                vkCmdDraw(commandBuffer, 3, 1, 0, 0);
            }

        protected:
            ~FabricatedTriangleDraw() override = default;
        };

        resolveStateGroup->addChild(FabricatedTriangleDraw::create());

        // Create the view to receive the command graph construction
        auto view{vsg::View::create(camera)};

        /*
         * Command graph construction
         */

        auto geomRenderGraph{vsg::RenderGraph::create()};
        geomRenderGraph->renderArea.offset = {0, 0};
        geomRenderGraph->renderArea.extent = window->extent2D();
        geomRenderGraph->framebuffer = geomFramebuffer;
        geomRenderGraph->addChild(geomStateGroup);
        view->addChild(geomRenderGraph);

        // create a pipeline barrier command to transition the headIndexImage's to the resolve pass
        {
            auto headIndexImageBarrier{vsg::ImageMemoryBarrier::create()};
            headIndexImageBarrier->srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            headIndexImageBarrier->dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            headIndexImageBarrier->oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            headIndexImageBarrier->newLayout = VK_IMAGE_LAYOUT_GENERAL;
            headIndexImageBarrier->image = headIndexImage;
            headIndexImageBarrier->subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            headIndexImageBarrier->subresourceRange.baseArrayLayer = 0;
            headIndexImageBarrier->subresourceRange.layerCount = 1;
            headIndexImageBarrier->subresourceRange.levelCount = 1;

            view->addChild(vsg::PipelineBarrier::create(
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, headIndexImageBarrier));
        }

        auto resolveRenderGraph{vsg::createRenderGraphForView(window, camera, resolveStateGroup)};
        resolveRenderGraph->setClearValues({{0.95f, 0.95f, 0.95f, 1.0f}}, {1.0f, 0});
        view->addChild(resolveRenderGraph);

        auto commandGraph{vsg::CommandGraph::create(window)};
        commandGraph->addChild(view);
        viewer->assignRecordAndSubmitTaskAndPresentation({commandGraph});
        viewer->compile();

        while (viewer->advanceToNextFrame())
        {
            viewer->handleEvents();
            viewer->update();
            viewer->recordAndSubmit();
            viewer->present();
        }
    }
    catch (const vsg::Exception& ve)
    {
        // details of VkResult values: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkResult.html
        for (int i = 0; i < argc; ++i) std::cerr << argv[i] << " ";
        std::cerr << "\n[Exception] - " << ve.message << " result = " << ve.result << std::endl;
        return 1;
    }
}
