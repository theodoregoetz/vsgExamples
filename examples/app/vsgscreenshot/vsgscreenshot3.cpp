/* <editor-fold desc="MIT License">

Copyright(c) 2018 Robert Osfield
Copyright(c) 2020 Tim Moore

Portions derived from code that is Copyright (C) Sascha Willems - www.saschawillems.de

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

</editor-fold> */

#include <vsg/all.h>

#ifdef vsgXchange_FOUND
#    include <vsgXchange/all.h>
#    include <vsgXchange/images.h>
#endif

#include <cassert>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

bool supportsBlit(vsg::ref_ptr<vsg::Device> device, VkFormat format)
{
    auto physicalDevice = device->getPhysicalDevice();
    VkFormatProperties srcFormatProperties;
    vkGetPhysicalDeviceFormatProperties(*(physicalDevice), format, &srcFormatProperties);
    VkFormatProperties destFormatProperties;
    vkGetPhysicalDeviceFormatProperties(*(physicalDevice), VK_FORMAT_R8G8B8A8_UNORM, &destFormatProperties);
    return ((srcFormatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) != 0) &&
           ((destFormatProperties.linearTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0);
}

vsg::ref_ptr<vsg::Image> createCaptureImage(
    vsg::ref_ptr<vsg::Device> device,
    VkFormat sourceFormat,
    const VkExtent2D& extent)
{
    // blit to RGBA if supported
    auto targetFormat = supportsBlit(device, sourceFormat) ? VK_FORMAT_R8G8B8A8_UNORM : sourceFormat;

    // create image to write to
    auto image = vsg::Image::create();
    image->format = targetFormat;
    image->extent = {extent.width, extent.height, 1};
    image->arrayLayers = 1;
    image->mipLevels = 1;
    image->tiling = VK_IMAGE_TILING_LINEAR;
    image->usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    image->compile(device);

    auto memReqs = image->getMemoryRequirements(device->deviceID);
    auto memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    auto deviceMemory = vsg::DeviceMemory::create(device, memReqs, memFlags);
    image->bind(deviceMemory, 0);

    return image;
}

VkImageUsageFlags computeUsageFlagsForFormat(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
        return VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    default:
        return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
}

vsg::ref_ptr<vsg::ImageView> createTransferImageView(
    vsg::ref_ptr<vsg::Device> device,
    VkFormat format,
    const VkExtent2D& extent,
    VkSampleCountFlagBits samples)
{
    auto image = vsg::Image::create();
    image->format = format;
    image->extent = VkExtent3D{extent.width, extent.height, 1};
    image->mipLevels = 1;
    image->arrayLayers = 1;
    image->samples = samples;
    image->usage = computeUsageFlagsForFormat(format) | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    image->initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

#if 0
/*
 * Somehow, we need to convert the layout of this image to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
 * If it's part of the commands in createTransferCommands(), we can convert it but the
 * screen capture breaks and the validation layer complains that it's trying to convert
 * from VK_IMAGE_LAYOUT_PREINITIALIZED to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL over and over.
 * Here, maybe the image layout can be converted once and ahead of time using a WaitEvent,
 * but it's unclear how to execute this outside of the viewer's main loop - it's like we
 * need a stand-alone commandgraph which is created, executes once and then is destroyed,
 * leaving the image to be used by the viewer's command graph later in the program.
 */

    {
        auto queueFamily = device->getPhysicalDevice()->getQueueFamily(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT);

        // ensure image attachments are setup on GPU.
        auto commandPool = vsg::CommandPool::create(device, graphicsFamily);
        vsg::submitCommandsToQueue(commandPool, device->getQueue(queueFamily), [&](vsg::CommandBuffer& commandBuffer) {
            auto imageBarrier = vsg::ImageMemoryBarrier::create(
                VK_ACCESS_MEMORY_READ_BIT,                                     // srcAccessMask
                VK_ACCESS_TRANSFER_READ_BIT,                                   // dstAccessMask
                VK_IMAGE_LAYOUT_UNDEFINED,                                // oldLayout
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,                          // newLayout
                VK_QUEUE_FAMILY_IGNORED,                                       // srcQueueFamilyIndex
                VK_QUEUE_FAMILY_IGNORED,                                       // dstQueueFamilyIndex
                image,                                                   // image
                VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1} // subresourceRange
            );
            auto pipelineBarrier = vsg::PipelineBarrier::create(
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                imageBarrier
            );
            pipelineBarrier->record(commandBuffer);
        });
    }
#endif

    return vsg::createImageView(device, image, vsg::computeAspectFlagsForFormat(format));
}

vsg::ref_ptr<vsg::Commands> createTransferCommands(
    vsg::ref_ptr<vsg::Device> device,
    vsg::ref_ptr<vsg::Image> sourceImage,
    vsg::ref_ptr<vsg::Image> destinationImage)
{
    auto commands = vsg::Commands::create();

    // transition destinationImage to transfer destination initialLayout
    auto transitionDestinationImageToDestinationLayoutBarrier = vsg::ImageMemoryBarrier::create(
        0,                                                             // srcAccessMask
        VK_ACCESS_TRANSFER_WRITE_BIT,                                  // dstAccessMask
        VK_IMAGE_LAYOUT_UNDEFINED,                                     // oldLayout
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,                          // newLayout
        VK_QUEUE_FAMILY_IGNORED,                                       // srcQueueFamilyIndex
        VK_QUEUE_FAMILY_IGNORED,                                       // dstQueueFamilyIndex
        destinationImage,                                              // image
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1} // subresourceRange
    );

    auto cmd_transitionForTransferBarrier = vsg::PipelineBarrier::create(
        VK_PIPELINE_STAGE_TRANSFER_BIT,                      // srcStageMask
        VK_PIPELINE_STAGE_TRANSFER_BIT,                      // dstStageMask
        0,                                                   // dependencyFlags
        transitionDestinationImageToDestinationLayoutBarrier // barrier
    );

    commands->addChild(cmd_transitionForTransferBarrier);

    if (
        sourceImage->format == destinationImage->format
        && sourceImage->extent.width == destinationImage->extent.width
        && sourceImage->extent.height == destinationImage->extent.height)
    {
        // use vkCmdCopyImage
        VkImageCopy region{};
        region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.srcSubresource.layerCount = 1;
        region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.dstSubresource.layerCount = 1;
        region.extent = destinationImage->extent;

        auto copyImage = vsg::CopyImage::create();
        copyImage->srcImage = sourceImage;
        copyImage->srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        copyImage->dstImage = destinationImage;
        copyImage->dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        copyImage->regions.push_back(region);

        commands->addChild(copyImage);
    }
    else if (supportsBlit(device, destinationImage->format))
    {
        // blit using vkCmdBlitImage
        VkImageBlit region{};
        region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.srcSubresource.layerCount = 1;
        region.srcOffsets[0] = VkOffset3D{0, 0, 0};
        region.srcOffsets[1] = VkOffset3D{
            static_cast<int32_t>(sourceImage->extent.width),
            static_cast<int32_t>(sourceImage->extent.height),
            1};
        region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.dstSubresource.layerCount = 1;
        region.dstOffsets[0] = VkOffset3D{0, 0, 0};
        region.dstOffsets[1] = VkOffset3D{
            static_cast<int32_t>(destinationImage->extent.width),
            static_cast<int32_t>(destinationImage->extent.height),
            1};

        auto blitImage = vsg::BlitImage::create();
        blitImage->srcImage = sourceImage;
        blitImage->srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        blitImage->dstImage = destinationImage;
        blitImage->dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        blitImage->regions.push_back(region);
        blitImage->filter = VK_FILTER_NEAREST;

        commands->addChild(blitImage);
    }
    else
    {
        /// If the source and target extents and/or format are different
        /// we would need to blit, however this device does not support it.
        /// Options at this point include resizing on the CPU (using STBI
        /// for example) or using a sampler.
        throw std::runtime_error{"GPU does not support blit."};
    }

    // transition destination image from transfer destination layout
    // to general layout to enable mapping to image DeviceMemory
    auto transitionDestinationImageToMemoryReadBarrier = vsg::ImageMemoryBarrier::create(
        VK_ACCESS_TRANSFER_WRITE_BIT,                                  // srcAccessMask
        VK_ACCESS_MEMORY_READ_BIT,                                     // dstAccessMask
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,                          // oldLayout
        VK_IMAGE_LAYOUT_GENERAL,                                       // newLayout
        VK_QUEUE_FAMILY_IGNORED,                                       // srcQueueFamilyIndex
        VK_QUEUE_FAMILY_IGNORED,                                       // dstQueueFamilyIndex
        destinationImage,                                              // image
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1} // subresourceRange
    );

    auto cmd_transitionFromTransferBarrier = vsg::PipelineBarrier::create(
        VK_PIPELINE_STAGE_TRANSFER_BIT,                // srcStageMask
        VK_PIPELINE_STAGE_TRANSFER_BIT,                // dstStageMask
        0,                                             // dependencyFlags
        transitionDestinationImageToMemoryReadBarrier  // barrier
    );

    commands->addChild(cmd_transitionFromTransferBarrier);

    return commands;
}

vsg::ref_ptr<vsg::RenderPass> createTransferRenderPass(
    vsg::ref_ptr<vsg::Device> device,
    VkFormat imageFormat,
    VkFormat depthFormat,
    bool requiresDepthRead)
{
    auto colorAttachment = vsg::defaultColorAttachment(imageFormat);
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; // <-- difference from vsg::createRenderPass()

    auto depthAttachment = vsg::defaultDepthAttachment(depthFormat);

    if (requiresDepthRead)
    {
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    }

    vsg::RenderPass::Attachments attachments{colorAttachment, depthAttachment};

    vsg::AttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    vsg::AttachmentReference depthAttachmentRef = {};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    vsg::SubpassDescription subpass = {};
    subpass.colorAttachments.emplace_back(colorAttachmentRef);
    subpass.depthStencilAttachments.emplace_back(depthAttachmentRef);

    vsg::RenderPass::Subpasses subpasses{subpass};

    // image layout transition
    vsg::SubpassDependency colorDependency = {};
    colorDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    colorDependency.dstSubpass = 0;
    colorDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    colorDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    colorDependency.srcAccessMask = 0;
    colorDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    colorDependency.dependencyFlags = 0;

    // depth buffer is shared between swap chain images
    vsg::SubpassDependency depthDependency = {};
    depthDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    depthDependency.dstSubpass = 0;
    depthDependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    depthDependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    depthDependency.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    depthDependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    depthDependency.dependencyFlags = 0;

    vsg::RenderPass::Dependencies dependencies{colorDependency, depthDependency};

    return vsg::RenderPass::create(device, attachments, subpasses, dependencies);
}

vsg::ref_ptr<vsg::RenderPass> createTransferRenderPass(
    vsg::ref_ptr<vsg::Device> device,
    VkFormat imageFormat,
    VkFormat depthFormat,
    VkSampleCountFlagBits samples,
    bool requiresDepthRead)
{
    vsg::ref_ptr<vsg::RenderPass> renderPass;

    if (samples == VK_SAMPLE_COUNT_1_BIT)
    {
        return createTransferRenderPass(device, imageFormat, depthFormat, requiresDepthRead);
    }

    // First attachment is multisampled target.
    vsg::AttachmentDescription colorAttachment = {};
    colorAttachment.format = imageFormat;
    colorAttachment.samples = samples;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // Second attachment is the resolved image which will be presented.
    vsg::AttachmentDescription resolveAttachment = {};
    resolveAttachment.format = imageFormat;
    resolveAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    resolveAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    resolveAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    resolveAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    resolveAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    resolveAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    resolveAttachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; // <-- difference from vsg::createMultisampledRenderPass()

    // multisampled depth attachment. Resolved if requiresDepthRead is true.
    vsg::AttachmentDescription depthAttachment = {};
    depthAttachment.format = depthFormat;
    depthAttachment.samples = samples;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    vsg::RenderPass::Attachments attachments{colorAttachment, resolveAttachment, depthAttachment};

    if (requiresDepthRead)
    {
        vsg::AttachmentDescription depthResolveAttachment = {};
        depthResolveAttachment.format = depthFormat;
        depthResolveAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthResolveAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthResolveAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthResolveAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthResolveAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthResolveAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthResolveAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachments.push_back(depthResolveAttachment);
    }

    vsg::AttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    vsg::AttachmentReference resolveAttachmentRef = {};
    resolveAttachmentRef.attachment = 1;
    resolveAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    vsg::AttachmentReference depthAttachmentRef = {};
    depthAttachmentRef.attachment = 2;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    vsg::SubpassDescription subpass;
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachments.emplace_back(colorAttachmentRef);
    subpass.resolveAttachments.emplace_back(resolveAttachmentRef);
    subpass.depthStencilAttachments.emplace_back(depthAttachmentRef);

    if (requiresDepthRead)
    {
        vsg::AttachmentReference depthResolveAttachmentRef = {};
        depthResolveAttachmentRef.attachment = 3;
        depthResolveAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        subpass.depthResolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
        subpass.stencilResolveMode = VK_RESOLVE_MODE_NONE;
        subpass.depthStencilResolveAttachments.emplace_back(depthResolveAttachmentRef);
    }

    vsg::RenderPass::Subpasses subpasses{subpass};

    vsg::SubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependency.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    vsg::SubpassDependency dependency2 = {};
    dependency2.srcSubpass = 0;
    dependency2.dstSubpass = VK_SUBPASS_EXTERNAL;
    dependency2.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency2.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependency2.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependency2.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependency2.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    vsg::RenderPass::Dependencies dependencies{dependency, dependency2};

    return vsg::RenderPass::create(device, attachments, subpasses, dependencies);
}

vsg::ref_ptr<vsg::Framebuffer> createOffscreenFramebuffer(
    vsg::ref_ptr<vsg::Device> device,
    vsg::ref_ptr<vsg::ImageView> transferImageView,
    VkSampleCountFlagBits const samples)
{
    constexpr VkFormat imageFormat = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
    constexpr bool requiresDepthRead = false;

    VkExtent2D const extent{
        transferImageView->image->extent.width,
        transferImageView->image->extent.height};

    vsg::ImageViews imageViews;
    if (samples == VK_SAMPLE_COUNT_1_BIT)
    {
        imageViews.emplace_back(transferImageView);
        imageViews.emplace_back(createTransferImageView(device, depthFormat, extent, VK_SAMPLE_COUNT_1_BIT));
    }
    else
    {
        // MSAA
        imageViews.emplace_back(createTransferImageView(device, imageFormat, extent, samples));
        imageViews.emplace_back(transferImageView);
        imageViews.emplace_back(createTransferImageView(device, depthFormat, extent, samples));
        if (requiresDepthRead)
        {
            imageViews.emplace_back(createTransferImageView(device, depthFormat, extent, VK_SAMPLE_COUNT_1_BIT));
        }
    }

    auto renderPass = createTransferRenderPass(device, imageFormat, depthFormat, samples, requiresDepthRead);
    auto framebuffer = vsg::Framebuffer::create(renderPass, imageViews, extent.width, extent.height, 1);

    return framebuffer;
}

vsg::ref_ptr<vsg::Camera> createCameraForScene(vsg::Node* scenegraph, const VkExtent2D& extent)
{
    // compute the bounds of the scene graph to help position camera
    vsg::ComputeBounds computeBounds;
    scenegraph->accept(computeBounds);
    vsg::dvec3 centre = (computeBounds.bounds.min + computeBounds.bounds.max) * 0.5;
    double radius = vsg::length(computeBounds.bounds.max - computeBounds.bounds.min) * 0.6;
    double nearFarRatio = 0.001;

    // set up the camera
    auto eye = centre + vsg::dvec3(0.0, -radius * 3.5, 0.0);
    auto target = centre;
    auto up = vsg::dvec3(0.0, 0.0, 1.0);
    auto lookAt = vsg::LookAt::create(eye, target, up);

    double fieldOfViewY = 30.0;
    double aspectRatio = static_cast<double>(extent.width) / static_cast<double>(extent.height);
    double nearDistance = nearFarRatio * radius;
    double farDistance = radius * 4.5;
    auto perspective = vsg::Perspective::create(fieldOfViewY, aspectRatio, nearDistance, farDistance);

    auto camera = vsg::Camera::create(perspective, lookAt, vsg::ViewportState::create(extent));
    return camera;
}

bool replaceChild(vsg::Group* group, vsg::ref_ptr<vsg::Node> previous, vsg::ref_ptr<vsg::Node> replacement)
{
    bool replaced = false;
    for (auto& child : group->children)
    {
        if (child == previous)
        {
            child = replacement;
            replaced = true;
        }
    }
    assert(replaced);
    return replaced;
}

class OffscreenCommandGraph : public vsg::Inherit<vsg::CommandGraph, OffscreenCommandGraph>
{
  public:
    OffscreenCommandGraph(vsg::ref_ptr<vsg::Device> in_device, int in_queueFamily, vsg::ref_ptr<vsg::Viewer> in_viewer, VkExtent2D const& in_extent);
    OffscreenCommandGraph(vsg::ref_ptr<vsg::Window> in_window, vsg::ref_ptr<vsg::Viewer> in_viewer);

    void setImageCapture(VkExtent2D const& in_extent, VkSampleCountFlagBits const in_samples, VkFormat const in_format);
    void setView(vsg::ref_ptr<vsg::View> in_view); // TODO: allow for multiple views
    void setEnabled(bool in_enabled);

    VkSampleCountFlagBits samples() const;

    vsg::ref_ptr<vsg::Data> getImageData();
    void saveImage(vsg::Path const& filename);

    vsg::ref_ptr<vsg::Viewer> viewer;
    vsg::ref_ptr<vsg::RenderGraph> renderGraph = vsg::RenderGraph::create();
    vsg::ref_ptr<vsg::Switch> renderSwitch = vsg::Switch::create();
    vsg::ref_ptr<vsg::Image> captureImage;
    vsg::ref_ptr<vsg::Commands> captureCommands;
    vsg::ref_ptr<vsg::View> view; // TODO: allow for multiple views
    bool enabled = false;

private:
    void init(VkExtent2D const& extent);
};

OffscreenCommandGraph::OffscreenCommandGraph(vsg::ref_ptr<vsg::Device> in_device, int in_queueFamily, vsg::ref_ptr<vsg::Viewer> in_viewer, VkExtent2D const& in_extent)
: Inherit{in_device, in_queueFamily}
, viewer{in_viewer}
{
    this->init(in_extent);
}

OffscreenCommandGraph::OffscreenCommandGraph(
    vsg::ref_ptr<vsg::Window> in_window,
    vsg::ref_ptr<vsg::Viewer> in_viewer)
: Inherit{in_window}
, viewer{in_viewer}
{
    this->init(window->extent2D());
}

void OffscreenCommandGraph::init(VkExtent2D const& extent)
{
    /// defaults
    constexpr VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;

    auto transferImageView = createTransferImageView(device, format, extent, samples);
    renderGraph->framebuffer = createOffscreenFramebuffer(device, transferImageView, samples);
    renderGraph->renderArea.extent = extent;
    renderGraph->setClearValues(
        VkClearColorValue{{0.0f, 0.0f, 0.0f, 0.0f}},
        VkClearDepthStencilValue{0.0f, 0});
    renderSwitch->addChild(enabled, renderGraph);
    this->addChild(renderSwitch);

    captureImage = createCaptureImage(device, format, extent);
    captureCommands = createTransferCommands(device, transferImageView->image, captureImage);
    this->addChild(captureCommands);
}

void OffscreenCommandGraph::setImageCapture(VkExtent2D const& extent, VkSampleCountFlagBits  samples, VkFormat  format)
{
    assert(captureImage);
    assert(renderGraph);
    assert(view);
    assert(view->camera);
    assert(view->camera->viewportState);
    assert(device);
    assert(captureCommands);
    if (extent.width != captureImage->extent.width
        || extent.height != captureImage->extent.height
        || samples != this->samples()
        || format != captureImage->format)
    {
        // TODO: better to scale according to viewport of display
        // to handle multiple views in the same render command graph
        view->camera->viewportState->set(0, 0, extent.width, extent.height);

        auto transferImageView = createTransferImageView(device, format, extent, VK_SAMPLE_COUNT_1_BIT);
        captureImage = createCaptureImage(device, format, extent);

        auto prevCaptureCommands = captureCommands;
        captureCommands = createTransferCommands(device, transferImageView->image, captureImage);
        replaceChild(this, prevCaptureCommands, captureCommands);

        renderGraph->framebuffer = createOffscreenFramebuffer(device, transferImageView, samples);
        renderGraph->resized();
        vsg::info("offscreen render resized to: ", extent.width, "x", extent.height);
    }
    assert(captureImage);
    assert(captureCommands);
    assert(renderGraph->framebuffer);
}

// TODO: allow for multiple views
void OffscreenCommandGraph::setView(vsg::ref_ptr<vsg::View> in_view)
{
    assert(in_view);
    assert(renderGraph);
    if (view)
    {
        replaceChild(renderGraph, view, in_view);
    }
    else
    {
        renderGraph->addChild(in_view);
        view = in_view;
    }
}

void OffscreenCommandGraph::setEnabled(bool in_enabled)
{
    assert(renderSwitch);
    enabled = in_enabled;
    renderSwitch->setAllChildren(enabled);
}

VkSampleCountFlagBits OffscreenCommandGraph::samples() const
{
    assert(renderGraph);
    assert(renderGraph->framebuffer);
    assert(renderGraph->framebuffer->getAttachments().size() > 0);
    assert(renderGraph->framebuffer->getAttachments().at(0));
    assert(renderGraph->framebuffer->getAttachments().at(0)->image);
    return renderGraph->framebuffer->getAttachments().at(0)->image->samples;
}

vsg::ref_ptr<vsg::Data> OffscreenCommandGraph::getImageData()
{
    assert(viewer);
    assert(device);
    assert(captureImage);

    constexpr uint64_t waitTimeout = 1000000000; // 1 second
    viewer->waitForFences(0, waitTimeout);

    VkImageSubresource subResource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
    VkSubresourceLayout subResourceLayout;
    vkGetImageSubresourceLayout(*device, captureImage->vk(device->deviceID), &subResource, &subResourceLayout);

    auto deviceMemory = captureImage->getDeviceMemory(device->deviceID);

    size_t destRowWidth = captureImage->extent.width * sizeof(vsg::ubvec4);
    vsg::ref_ptr<vsg::Data> imageData;
    if (destRowWidth == subResourceLayout.rowPitch)
    {
        /// Map the buffer memory and assign as a vec4Array2D that will automatically
        /// unmap itself on destruction.
        imageData = vsg::MappedData<vsg::ubvec4Array2D>::create(
            deviceMemory,
            subResourceLayout.offset,
            0,
            vsg::Data::Properties{captureImage->format},
            captureImage->extent.width,
            captureImage->extent.height);
    }
    else
    {
        /// Map the buffer memory and assign as a ubyteArray that will automatically
        /// unmap itself on destruction. A ubyteArray is used as the graphics buffer
        /// memory is not contiguous like vsg::Array2D, so map to a flat buffer first
        /// then copy to Array2D.
        auto mappedData = vsg::MappedData<vsg::ubyteArray>::create(
            deviceMemory,
            subResourceLayout.offset,
            0,
            vsg::Data::Properties{captureImage->format},
            subResourceLayout.rowPitch * captureImage->extent.height);
        imageData = vsg::ubvec4Array2D::create(captureImage->extent.width, captureImage->extent.height, vsg::Data::Properties{captureImage->format});
        for (uint32_t row = 0; row < captureImage->extent.height; ++row)
        {
            std::memcpy(imageData->dataPointer(row * captureImage->extent.width), mappedData->dataPointer(row * subResourceLayout.rowPitch), destRowWidth);
        }
    }
    return imageData;
}

void OffscreenCommandGraph::saveImage(vsg::Path const& filename)
{
    vsg::info("writing image to file: ", filename);
    auto imageData = this->getImageData();
    auto options = vsg::Options::create();
    options->add(vsgXchange::stbi::create());
    vsg::write(imageData, filename, options);
    vsg::info("image saved.");
}

class DisplayView : public vsg::Inherit<vsg::View, DisplayView>
{
public:
    DisplayView(vsg::ref_ptr<vsg::Camera> in_camera, vsg::ref_ptr<vsg::Node> in_scenegraph = {});
    void setOffscreenExtent(VkExtent2D const& extent);
    VkExtent2D syncOffscreenExtent();
    void syncOffscreenFieldOfView();

    vsg::ref_ptr<vsg::View> offscreenView;
};

DisplayView::DisplayView(vsg::ref_ptr<vsg::Camera> in_camera, vsg::ref_ptr<vsg::Node> in_scenegraph)
: Inherit{in_camera, in_scenegraph}
{
    auto offscreenCamera = vsg::Camera::create();
    offscreenCamera->viewMatrix = camera->viewMatrix;
    offscreenCamera->projectionMatrix = vsg::Perspective::create();
    offscreenCamera->viewportState = vsg::ViewportState::create();
    offscreenView = vsg::View::create(offscreenCamera, in_scenegraph);
    this->syncOffscreenExtent();
    this->syncOffscreenFieldOfView();
}

void DisplayView::setOffscreenExtent(VkExtent2D const& extent)
{
    assert(offscreenView);
    assert(offscreenView->camera);
    assert(offscreenView->camera->projectionMatrix);
    assert(offscreenView->camera->viewportState);

    auto offscreenPerspective = dynamic_cast<vsg::Perspective*>(offscreenView->camera->projectionMatrix.get());
    offscreenPerspective->aspectRatio = static_cast<double>(extent.width) / static_cast<double>(extent.height);
    offscreenView->camera->viewportState->set(0, 0, extent.width, extent.height);
}

VkExtent2D DisplayView::syncOffscreenExtent()
{
    assert(camera);

    auto extent = camera->getRenderArea().extent;
    this->setOffscreenExtent(extent);
    return extent;
}

void DisplayView::syncOffscreenFieldOfView()
{
    assert(camera);
    assert(camera->projectionMatrix);
    assert(offscreenView);
    assert(offscreenView->camera);
    assert(offscreenView->camera->projectionMatrix);

    auto perspective = dynamic_cast<vsg::Perspective*>(camera->projectionMatrix.get());
    auto offscreenPerspective = dynamic_cast<vsg::Perspective*>(offscreenView->camera->projectionMatrix.get());
    offscreenPerspective->fieldOfViewY = perspective->fieldOfViewY;
    offscreenPerspective->nearDistance = perspective->nearDistance;
    offscreenPerspective->farDistance = perspective->farDistance;
}

class DisplayViewer : public vsg::Inherit<vsg::Viewer, DisplayViewer>
{
public:
    DisplayViewer(vsg::ref_ptr<vsg::Window> window);
    void setView(vsg::ref_ptr<DisplayView> in_view);
    void syncImageCaptureToDisplay();
    void setImageCapture(VkExtent2D const& extent, VkSampleCountFlagBits samples, VkFormat format);
    void saveImage(vsg::Path const& filename);

    vsg::ref_ptr<vsg::Window> window;
    vsg::ref_ptr<DisplayView> displayView;
    vsg::ref_ptr<OffscreenCommandGraph> offscreenCommandGraph;
};

DisplayViewer::DisplayViewer(vsg::ref_ptr<vsg::Window> in_window)
: Inherit{}
, window{in_window}
{
    this->addWindow(window);
}

void DisplayViewer::setView(vsg::ref_ptr<DisplayView> in_view)
{
    displayView = in_view;
    auto displayRenderGraph = vsg::RenderGraph::create(window, displayView);
    auto displayCommandGraph = vsg::CommandGraph::create(window);
    displayCommandGraph->addChild(displayRenderGraph);
    offscreenCommandGraph = OffscreenCommandGraph::create(window, vsg::ref_ptr{this});
    offscreenCommandGraph->setView(displayView->offscreenView);
    this->assignRecordAndSubmitTaskAndPresentation({offscreenCommandGraph, displayCommandGraph});
    this->compile();
}

void DisplayViewer::syncImageCaptureToDisplay()
{
    assert(displayView);
    assert(offscreenCommandGraph);
    assert(offscreenCommandGraph->captureImage);
    auto extent = displayView->syncOffscreenExtent();
    offscreenCommandGraph->setImageCapture(extent, offscreenCommandGraph->samples(), offscreenCommandGraph->captureImage->format);
}

void DisplayViewer::setImageCapture(VkExtent2D const& extent, VkSampleCountFlagBits samples, VkFormat format)
{
    assert(displayView);
    assert(offscreenCommandGraph);
    displayView->setOffscreenExtent(extent);
    offscreenCommandGraph->setImageCapture(extent, samples, format);
    assert(offscreenCommandGraph->captureImage);
}

void DisplayViewer::saveImage(vsg::Path const& filename)
{
    assert(displayView);
    assert(offscreenCommandGraph);
    displayView->syncOffscreenFieldOfView();
    offscreenCommandGraph->setEnabled(true);
    this->update();
    this->recordAndSubmit();
    this->present();
    offscreenCommandGraph->saveImage(filename);
    offscreenCommandGraph->setEnabled(false);
}

std::tuple<vsg::ref_ptr<vsg::Device>, int> createOffscreenDevice()
{
    auto vulkanVersion = VK_API_VERSION_1_2;
    bool debugLayer = true;
    vsg::Names instanceExtensions;
    vsg::Names requestedLayers;
    if (debugLayer)
    {
        instanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        instanceExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        requestedLayers.push_back("VK_LAYER_KHRONOS_validation");
    }

    vsg::Names validatedNames = vsg::validateInstancelayerNames(requestedLayers);

    auto instance = vsg::Instance::create(instanceExtensions, validatedNames, vulkanVersion);
    auto [physicalDevice, queueFamily] = instance->getPhysicalDeviceAndQueueFamily(VK_QUEUE_GRAPHICS_BIT);
    if (!physicalDevice || queueFamily < 0)
    {
        std::cout << "Could not create PhysicalDevice" << std::endl;
        throw std::runtime_error("Could not create PhysicalDevice");
    }

    vsg::Names deviceExtensions;
    vsg::QueueSettings queueSettings{vsg::QueueSetting{queueFamily, {1.0}}};

    bool enableGeometryShader = true;
    auto deviceFeatures = vsg::DeviceFeatures::create();
    deviceFeatures->get().samplerAnisotropy = VK_TRUE;
    deviceFeatures->get().geometryShader = enableGeometryShader;

    auto device = vsg::Device::create(physicalDevice, queueSettings, validatedNames, deviceExtensions, deviceFeatures);

    return std::tie(device, queueFamily);
}

class HeadlessViewer : public vsg::Inherit<vsg::Viewer, HeadlessViewer>
{
public:
    HeadlessViewer(VkExtent2D const& extent);
    void setView(vsg::ref_ptr<vsg::View> view); // TODO: allow for multiple views
    void saveImage(vsg::Path const& filename);

    vsg::ref_ptr<OffscreenCommandGraph> offscreenCommandGraph;
};

HeadlessViewer::HeadlessViewer(VkExtent2D const& extent)
: Inherit{}
{
    auto [device, queueFamily] = createOffscreenDevice();
    offscreenCommandGraph = OffscreenCommandGraph::create(device, queueFamily, vsg::ref_ptr(this), extent);
    offscreenCommandGraph->setEnabled(true);
    this->assignRecordAndSubmitTaskAndPresentation({offscreenCommandGraph});
}

// TODO: allow for multiple views
void HeadlessViewer::setView(vsg::ref_ptr<vsg::View> view)
{
    offscreenCommandGraph->setView(view);
    this->compile();
}

void HeadlessViewer::saveImage(vsg::Path const& filename)
{
    this->update();
    this->recordAndSubmit();
    this->present();
    offscreenCommandGraph->saveImage(filename);
}

class ScreenshotHandler : public vsg::Inherit<vsg::Visitor, ScreenshotHandler>
{
public:
    bool do_sync_extent = false;
    bool do_image_capture = false;

    ScreenshotHandler()
    {
        vsg::info("press 'f' to save offscreen render to file");
        vsg::info("press 'e' to set offscreen render extents to same as display");
    }

    void apply(vsg::KeyPressEvent& keyPress) override
    {
        switch (keyPress.keyBase)
        {
            case 'e': do_sync_extent = true; break;
            case 'f': do_image_capture = true; break;
            default: break;
        }
    }
};

int main(int argc, char** argv)
{
    // set up defaults and read command line arguments to override them
    vsg::CommandLine arguments(&argc, argv);

    auto windowTraits = vsg::WindowTraits::create();
    windowTraits->windowTitle = "screenshot3";
    windowTraits->debugLayer = arguments.read({"--debug", "-d"});
    windowTraits->apiDumpLayer = arguments.read({"--api", "-a"});
    windowTraits->synchronizationLayer = arguments.read("--sync");

    // offscreen capture filename and multi sampling parameters
    auto captureFilename = arguments.value<vsg::Path>("screenshot.vsgt", {"--capture-file", "-f"});
    bool msaa = arguments.read("--msaa");
    bool headless = arguments.read("--headless");

    if (arguments.errors()) return arguments.writeErrorMessages(std::cerr);

    // if we are multisampling then to enable copying of the depth buffer we have to
    // enable a depth buffer resolve extension for vsg::RenderPass or require a minimum vulkan version of 1.2
    if (msaa) windowTraits->vulkanVersion = VK_API_VERSION_1_2;

    // sampling for offscreen rendering
    VkSampleCountFlagBits samples = msaa ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_1_BIT;

    // read shaders
    vsg::Paths searchPaths = vsg::getEnvPaths("VSG_FILE_PATH");

    using VsgNodes = std::vector<vsg::ref_ptr<vsg::Node>>;
    VsgNodes vsgNodes;

    auto options = vsg::Options::create();
    options->fileCache = vsg::getEnv("VSG_FILE_CACHE");
    options->paths = vsg::getEnvPaths("VSG_FILE_PATH");

#ifdef vsgXchange_all
    // add vsgXchange's support for reading and writing 3rd party file formats
    options->add(vsgXchange::all::create());
#endif

    // read any vsg files
    for (int i = 1; i < argc; ++i)
    {
        vsg::Path filename = arguments[i];
        auto loaded_scene = vsg::read_cast<vsg::Node>(filename, options);
        if (loaded_scene)
        {
            vsgNodes.push_back(loaded_scene);
            arguments.remove(i, 1);
            --i;
        }
    }

    // assign the vsg_scene from the loaded nodes
    vsg::ref_ptr<vsg::Node> vsg_scene;
    if (vsgNodes.size() > 1)
    {
        auto vsg_group = vsg::Group::create();
        for (auto& subgraphs : vsgNodes)
        {
            vsg_group->addChild(subgraphs);
        }

        vsg_scene = vsg_group;
    }
    else if (vsgNodes.size() == 1)
    {
        vsg_scene = vsgNodes.front();
    }

    if (!vsg_scene)
    {
        std::cout << "No valid model files specified." << std::endl;
        return 1;
    }

    // Transform for rotation animation
    auto transform = vsg::MatrixTransform::create();
    transform->addChild(vsg_scene);
    vsg_scene = transform;

    if (headless)
    {
        VkExtent2D extent{800, 600};

        auto camera = createCameraForScene(vsg_scene, extent);
        auto view = vsg::View::create(camera, vsg_scene);

        auto viewer = HeadlessViewer::create(extent);
        viewer->setView(view);
        viewer->saveImage(captureFilename);
    }

    else
    {
        auto window = vsg::Window::create(windowTraits);
        if (!window)
        {
            std::cout << "Could not create window." << std::endl;
            return 1;
        }

        auto viewer = DisplayViewer::create(window);

        auto displayCamera = createCameraForScene(vsg_scene, window->extent2D());
        auto displayView = DisplayView::create(displayCamera, vsg_scene);
        viewer->setView(displayView);

        // add close handler to respond to the close window button and pressing escape
        viewer->addEventHandler(vsg::CloseHandler::create(viewer));
        viewer->addEventHandler(vsg::Trackball::create(displayCamera));

        auto screenshotHandler = ScreenshotHandler::create();
        viewer->addEventHandler(screenshotHandler);

        // rendering main loop
        while (viewer->advanceToNextFrame())
        {
            viewer->handleEvents();

            if (screenshotHandler->do_sync_extent)
            {
                screenshotHandler->do_sync_extent = false;
                viewer->syncImageCaptureToDisplay();
            }
            else if (screenshotHandler->do_image_capture)
            {
                screenshotHandler->do_image_capture = false;
                viewer->saveImage(captureFilename);
                continue;
            }

            viewer->update();
            viewer->recordAndSubmit();
            viewer->present();
        }
    }

    // clean up done automatically thanks to ref_ptr<>
    return 0;
}
