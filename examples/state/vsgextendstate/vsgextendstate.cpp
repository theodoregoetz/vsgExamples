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
#extension GL_ARB_separate_shader_objects : enable

void main()
{
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(uv * 2.0f + -1.0f, 0.0f, 1.0f);
}
)"};

std::string geomFragShaderSource{R"(
#version 450
#extension GL_ARB_separate_shader_objects : enable
#pragma import_defines ( OIT )

#define OIT 1
#if defined(OIT)
    layout (early_fragment_tests) in;

    struct Node
    {
        vec4 color;
        float depth;
        uint next;
    };

    layout (set = 0, binding = 1) buffer GeometrySBO
    {
        uint count;
        uint maxNodeCount;
    };

    layout (set = 0, binding = 2, r32ui) uniform coherent uimage2D headIndexImage;

    layout (set = 0, binding = 3) buffer LinkedListSBO
    {
        Node nodes[];
    };
#else
    layout(location = 0) out vec4 outFragColor;
#endif

void main() {
    float a = gl_PrimitiveID*0.1;
    float b = 1.0-gl_PrimitiveID*0.1;
    vec4 color;
    if (gl_PrimitiveID % 2 == 0)
        color = vec4(a, b, 0, 0.5);
    else
        color = vec4(b, a, 0, 0.5);

    #if defined(OIT)
        // Increase the node count
        uint nodeIdx = atomicAdd(count, 1);

        // Check LinkedListSBO is full
        if (nodeIdx < maxNodeCount)
        {
            // Exchange new head index and previous head index
            uint prevHeadIdx = imageAtomicExchange(headIndexImage, ivec2(gl_FragCoord.xy), nodeIdx);

            // Store node data
            nodes[nodeIdx].color = color;
            nodes[nodeIdx].depth = gl_FragCoord.z;
            nodes[nodeIdx].next = prevHeadIdx;
        }
    #else
        outFragColor = color;
    #endif
}
)"};

std::string resolveFragShaderSource{R"(
#version 450
#extension GL_ARB_separate_shader_objects : enable

#define MAX_FRAGMENT_COUNT 128

struct Node
{
    vec4 color;
    float depth;
    uint next;
};

layout (location = 0) out vec4 outFragColor;

layout (set = 0, binding = 0, r32ui) uniform uimage2D headIndexImage;

layout (set = 0, binding = 1) buffer LinkedListSBO
{
    Node nodes[];
};

void main()
{
    Node fragments[MAX_FRAGMENT_COUNT];
    int count = 0;

    uint nodeIdx = imageLoad(headIndexImage, ivec2(gl_FragCoord.xy)).r;

    while (nodeIdx != 0xffffffff && count < MAX_FRAGMENT_COUNT)
    {
        fragments[count] = nodes[nodeIdx];
        nodeIdx = fragments[count].next;
        ++count;
    }

    // Do the insertion sort
    for (uint i = 1; i < count; ++i)
    {
        Node insert = fragments[i];
        uint j = i;
        while (j > 0 && insert.depth > fragments[j - 1].depth)
        {
            fragments[j] = fragments[j-1];
            --j;
        }
        fragments[j] = insert;
    }

    // Do blending
    vec4 color = vec4(0.025, 0.025, 0.025, 1.0f);
    for (int i = 0; i < count; ++i)
    {
        color = mix(color, fragments[i].color, fragments[i].color.a);
    }

    outFragColor = color;
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

        int gpuNumber{1};
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

        // define a data structure for maintaining the linked list current and max node count
        struct LinkedListCurSize
        {
            uint32_t count;
            uint32_t maxNodeCount;
        };
#if 0
        auto stagingBuffer = vsg::createBufferAndMemory(
            window->getOrCreateDevice(),
            sizeof(LinkedListCurSize),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        auto stagingBufferMemory{stagingBuffer->getDeviceMemory(window->getOrCreateDevice())};
        VkResult map(VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** ppData);
        void* stagingBufferMappedMemory{};
        auto vkResult{stagingBufferMemory->map(0, sizeof(LinkedListCurSize), 0, &stagingBufferMappedMemory)};
        assert(vkResult == VK_SUCCESS);
#endif

        auto linkedListCurSizeBuffer = vsg::createBufferAndMemory(
            window->getOrCreateDevice(),
            sizeof(LinkedListCurSize),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        auto linkedListCurSizeInfo{vsg::BufferInfo::create()};
        linkedListCurSizeInfo->buffer = linkedListCurSizeBuffer;
        linkedListCurSizeInfo->offset = 0;
        linkedListCurSizeInfo->range = sizeof(LinkedListCurSize);

        // Set up GeometrySBO data.
        constexpr uint32_t MAX_FRAGMENT_NODE_COUNT{5};
        LinkedListCurSize linkedListCurSize{
            0,
            MAX_FRAGMENT_NODE_COUNT * window->extent2D().width * window->extent2D().height};
#if 0
        memcpy(stagingBufferMappedMemory, &linkedListCurSize, sizeof(LinkedListCurSize));

        // TODO: need to data from stagingBuffer to linkedListCurSizeBuffer
        stagingBuffer.reset(); // after copy we no longer need the staging buffer
#endif

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


        // define a fragment node data structure to collect in the geometry pass and sort in the resolve pass
        struct FragmentNode
        {
            vsg::vec4 color;
            float     depth;
            uint32_t  next;
        };

        // create a buffer for the fragment linked list
        auto linkedListBufferSize{static_cast<VkDeviceSize>(
            sizeof(FragmentNode) * linkedListCurSize.maxNodeCount)};
        auto linkedListBuffer{vsg::createBufferAndMemory(
            window->getOrCreateDevice(), linkedListBufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        auto linkedListInfo{vsg::BufferInfo::create()};
        linkedListInfo->buffer = linkedListBuffer;
        linkedListInfo->offset = 0;
        linkedListInfo->range = linkedListBufferSize;


        // create a barrier to change headIndexImage's layout from UNDEFINED to GENERAL
        auto headIndexImageBarrier{vsg::ImageMemoryBarrier::create()};
        headIndexImageBarrier->srcAccessMask = 0;
        headIndexImageBarrier->dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        headIndexImageBarrier->oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        headIndexImageBarrier->newLayout = VK_IMAGE_LAYOUT_GENERAL;
        headIndexImageBarrier->image = headIndexImage;
        headIndexImageBarrier->subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        headIndexImageBarrier->subresourceRange.baseArrayLayer = 0;
        headIndexImageBarrier->subresourceRange.layerCount = 1;
        headIndexImageBarrier->subresourceRange.levelCount = 1;

        // change headIndexImage's layout from UNDEFINED to GENERAL
        auto headIndexImageBarrierCommand{vsg::PipelineBarrier::create(
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, headIndexImageBarrier)};
        auto queueFamilyIndex{window->getOrCreatePhysicalDevice()->getQueueFamily(VK_QUEUE_GRAPHICS_BIT)};
        auto commandPool{vsg::CommandPool::create(window->getOrCreateDevice(), queueFamilyIndex)};
        auto queue{window->getOrCreateDevice()->getQueue(queueFamilyIndex)};
        vsg::submitCommandsToQueue(commandPool, nullptr, 0, queue,
            [&](vsg::CommandBuffer& commandBuffer) {
                headIndexImageBarrierCommand->record(commandBuffer);
            });

        // geometry pipeline construction
        vsg::DescriptorSetLayoutBindings geomDescriptorBindings{
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
        };
        auto geomLinkedListCurSizeInfoDesc{vsg::DescriptorBuffer::create(vsg::BufferInfoList{linkedListCurSizeInfo},
            1, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)};
        auto geomHeadIndexImageDesc{vsg::DescriptorImage::create(headIndexImageInfo,
            2, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)};
        auto geomLinkedListDesc{vsg::DescriptorBuffer::create(vsg::BufferInfoList{linkedListInfo},
            3, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)};

        auto geomDescriptorSetLayout{vsg::DescriptorSetLayout::create(geomDescriptorBindings)};
        auto geomDescriptorSet{vsg::DescriptorSet::create(geomDescriptorSetLayout,
            vsg::Descriptors{geomLinkedListCurSizeInfoDesc,geomHeadIndexImageDesc,geomLinkedListDesc})};

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
        depthStencilState->depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
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
            {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}
        };
        auto resolveHeadIndexImageDesc{vsg::DescriptorImage::create(headIndexImageInfo,
            0, 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)};
        auto resolveLinkedListDesc{vsg::DescriptorBuffer::create(vsg::BufferInfoList{linkedListInfo},
            1, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)};

        auto resolveDescriptorSetLayout{vsg::DescriptorSetLayout::create(resolveDescriptorBindings)};
        auto resolveDescriptorSet{vsg::DescriptorSet::create(resolveDescriptorSetLayout,
            vsg::Descriptors{resolveHeadIndexImageDesc,resolveLinkedListDesc})};

        vsg::GraphicsPipelineStates resolvePipelineStates{
            vsg::VertexInputState::create(),
            inputAssemblyState,
            rasterizationState,
            vsg::MultisampleState::create(),
            vsg::ColorBlendState::create(),
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

        /*
         * Command graph
         */
        auto commandGraph{vsg::CommandGraph::create(window)};

        // reset the headImage texture with end-of-list values, 0xffffffff
        VkClearColorValue headImageClearColor;
        headImageClearColor.uint32[0] = 0xffffffff;

        VkImageSubresourceRange subresRange{};
        subresRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresRange.levelCount = 1;
        subresRange.layerCount = 1;

        auto headImageClearColorCommand{vsg::ClearColorImage::create()};
        headImageClearColorCommand->image = headIndexImage;
        headImageClearColorCommand->imageLayout = headIndexImageInfo->imageLayout;
        headImageClearColorCommand->color = headImageClearColor;
        headImageClearColorCommand->ranges.push_back(subresRange);

        commandGraph->addChild(headImageClearColorCommand);

        // reset the linked list current size, count, to zero by setting just the first 4
        // bytes, leaving the second half of the buffer, maxNodeCount, untouched
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
#if 0
        commandGraph->addChild(FillBuffer::create(linkedListCurSizeBuffer, 0, sizeof(uint32_t), 0));
#else
        commandGraph->addChild(FillBuffer::create(linkedListCurSizeBuffer, 0, sizeof(uint32_t), 0));
        commandGraph->addChild(FillBuffer::create(linkedListCurSizeBuffer, sizeof(uint32_t), sizeof(uint32_t), linkedListCurSize.maxNodeCount));
#endif

        // create a memory barrier to ensure all previous writes are finished before we start to write again
        {
            auto memoryBarrier{vsg::MemoryBarrier::create()};
            memoryBarrier->srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            memoryBarrier->dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            auto memoryBarrierCommand{vsg::PipelineBarrier::create(
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, memoryBarrier)};
            commandGraph->addChild(memoryBarrierCommand);
        }

        auto view{vsg::View::create(camera)};
        view->addChild(geomStateGroup);

        auto geomRenderGraph{vsg::RenderGraph::create()};
        geomRenderGraph->renderArea.offset = {0, 0};
        geomRenderGraph->renderArea.extent = window->extent2D();
        geomRenderGraph->framebuffer = geomFramebuffer;
        geomRenderGraph->addChild(view);
        commandGraph->addChild(geomRenderGraph);

        // ensure geometry pass is complete before starting the resolve pass
        commandGraph->addChild(vsg::PipelineBarrier::create(
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0));

        // create a memory barrier to ensure all writes are finished before we start to write again
        {
            auto memoryBarrier{vsg::MemoryBarrier::create()};
            memoryBarrier->srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            memoryBarrier->dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            auto memoryBarrierCommand{vsg::PipelineBarrier::create(
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, memoryBarrier)};
            commandGraph->addChild(memoryBarrierCommand);
        }

        auto resolveRenderGraph{vsg::createRenderGraphForView(window, camera, resolveStateGroup)};
        resolveRenderGraph->setClearValues({{0.025f, 0.025f, 0.025f, 1.0f}}, {1.0f, 0});
        commandGraph->addChild(resolveRenderGraph);

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
