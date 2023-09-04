#include <string>

#include "vsg/all.h"

std::string VERT{R"(
#version 450
layout(push_constant) uniform PushConstants { mat4 projection; mat4 modelView; };
layout(location = 0) in vec3 vertex;
out gl_PerVertex { vec4 gl_Position; };
void main() { gl_Position = (projection * modelView) * vec4(vertex, 1.0); }
)"};

std::string FRAG{R"(
#version 450
layout(set = 0, binding = 0) buffer CellColors { vec4[] cellColors; };
layout(location = 0) out vec4 color;
void main() { color = cellColors[gl_PrimitiveID]; }
)"};


int main(int argc, char** argv)
{
    auto windowTraits = vsg::WindowTraits::create();
    windowTraits->windowTitle = "vsgstoragebuffer";

    // set up defaults and read command line arguments to override them
    vsg::CommandLine arguments(&argc, argv);
    windowTraits->debugLayer = arguments.read({"--debug", "-d"});
    windowTraits->apiDumpLayer = arguments.read({"--api", "-a"});

    auto vertexShader = vsg::ShaderStage::create(VK_SHADER_STAGE_VERTEX_BIT, "main", VERT);
    auto fragmentShader = vsg::ShaderStage::create(VK_SHADER_STAGE_FRAGMENT_BIT, "main", FRAG);
    auto shaderSet = vsg::ShaderSet::create(vsg::ShaderStages{vertexShader, fragmentShader});
    shaderSet->addPushConstantRange("pc", "", VK_SHADER_STAGE_VERTEX_BIT, 0, 128);

    shaderSet->addBufferBinding("cellColors", "", 0, 0,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT,
        vsg::vec4Array::create({{1,0,0,1}}));

    shaderSet->addAttributeBinding("vertex", "", 0, VK_FORMAT_R32G32B32_SFLOAT,
        vsg::vec3Array::create(1));

    auto sceneGraph = vsg::Group::create();
    auto stateGroup = vsg::StateGroup::create();
    auto gpConf = vsg::GraphicsPipelineConfigurator::create(shaderSet);

    /**
     * This array is used in the fragment shader to color each individual triangle
     * using gl_PrimitiveID as the array index. In principle, this could be
     * a very large array that would exceed the uniform buffer size limits.
     */
    auto cellColors = vsg::vec4Array::create({
        {0.176, 0.408, 0.376, 1.0},  // Powderkeg Blue (triangle 0)
        {0.949, 0.663, 0.000, 1.0},  // Westwood Gold  (triangle 1)
    });
    gpConf->assignBuffer("cellColors", cellColors);

    /// single quad
    auto vertices = vsg::vec3Array::create({
        {-1, 0, -1}, { 1, 0, -1}, { 1, 0,  1}, {-1, 0,  1}
    });
    auto indices = vsg::uintArray::create({0, 1, 2, 0, 2, 3});
    vsg::DataList vertexArrays;
    gpConf->assignArray(vertexArrays, "vertex", VK_VERTEX_INPUT_RATE_VERTEX, vertices);

    auto vertexDraw = vsg::VertexIndexDraw::create();
    vertexDraw->assignArrays(vertexArrays);
    vertexDraw->assignIndices(indices);
    vertexDraw->indexCount = static_cast<uint32_t>(indices->size());
    vertexDraw->instanceCount = 1;
    stateGroup->addChild(vertexDraw);

    gpConf->init();
    gpConf->copyTo(stateGroup);
    sceneGraph->addChild(stateGroup);

    auto window = vsg::Window::create(windowTraits);
    auto lookAt = vsg::LookAt::create(
        vsg::dvec3{2,-5,-1},
        vsg::dvec3{0, 0, 0},
        vsg::dvec3{0, 0, 1}
    );
    auto perspective = vsg::Perspective::create();
    auto viewportState = vsg::ViewportState::create(window->extent2D());
    auto camera = vsg::Camera::create(perspective, lookAt, viewportState);
    auto viewer = vsg::Viewer::create();

    viewer->addWindow(window);
    viewer->addEventHandler(vsg::CloseHandler::create(viewer));
    viewer->addEventHandler(vsg::Trackball::create(camera));

    auto commandGraph = vsg::createCommandGraphForView(window, camera, sceneGraph);
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
