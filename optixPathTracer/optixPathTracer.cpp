#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#    include <GL/freeglut.h>
#  else
#    include <GL/glut.h>
#  endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include "optixPathTracer.h"
#include <sutil.h>
#include <Arcball.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdint.h>

#include <OptiXMesh.h>

#define MY_PI 3.1415926f

using namespace optix;

const char* const SAMPLE_NAME = "optixPathTracer";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context        context = 0;
uint32_t       width  = 800;
uint32_t       height = 800;
bool           use_pbo = true;

int            frame_number = 1;
int            sqrt_num_samples = 16;
int            rr_begin_depth = 1;
Program        pgram_intersection = 0;
Program        pgram_bounding_box = 0;

// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
bool           camera_changed = true;
sutil::Arcball arcball;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

// Post-processing
CommandList commandListWithDenoiser;
CommandList commandListWithoutDenoiser;
PostprocessingStage tonemapStage;
PostprocessingStage denoiserStage;
Buffer denoisedBuffer;
Buffer emptyBuffer;
Buffer trainingDataBuffer;

// Defines the amount of the original image that is blended with the denoised result
// ranging from 0.0 to 1.0
float denoiseBlend = 0.75f;



//------------------------------------------------------------------------------
//
// Forward decls 
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void loadGeometry();
void setupCamera();
void updateCamera();
void glutInitialize( int* argc, char** argv );
void glutRun();

void glutDisplay();
void glutKeyboardPress( unsigned char k, int x, int y );
void glutMousePress( int button, int state, int x, int y );
void glutMouseMotion( int x, int y);
void glutResize( int w, int h );




Buffer getOutputBuffer()
{
    return context[ "output_buffer" ]->getBuffer();
}


void destroyContext()
{
    if( context )
    {
        context->destroy();
        context = 0;
    }
}


void registerExitHandler()
{
    // register shutdown handler
#ifdef _WIN32
    glutCloseFunc( destroyContext );  // this function is freeglut-only
#else
    atexit( destroyContext );
#endif
}


void setMaterial(
        GeometryInstance& gi,
        Material material,
        const std::string& color_name,
        const float3& color)
{
    gi->addMaterial(material);
    gi[color_name]->setFloat(color);
}


GeometryInstance createParallelogram(
        const float3& anchor,
        const float3& offset1,
        const float3& offset2)
{
    Geometry parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount( 1u );
    parallelogram->setIntersectionProgram( pgram_intersection );
    parallelogram->setBoundingBoxProgram( pgram_bounding_box );

    float3 normal = normalize( cross( offset1, offset2 ) );
    float d = dot( normal, anchor );
    float4 plane = make_float4( normal, d );

    float3 v1 = offset1 / dot( offset1, offset1 );
    float3 v2 = offset2 / dot( offset2, offset2 );

    parallelogram["plane"]->setFloat( plane );
    parallelogram["anchor"]->setFloat( anchor );
    parallelogram["v1"]->setFloat( v1 );
    parallelogram["v2"]->setFloat( v2 );

    GeometryInstance gi = context->createGeometryInstance();
    gi->setGeometry(parallelogram);
    return gi;
}


void createContext()
{
    context = Context::create();
    context->setRayTypeCount( 3 );
    context->setEntryPointCount( 1 );
    context->setStackSize( 1800 );

    context[ "scene_epsilon"                  ]->setFloat( 1.e-3f );
    context[ "pathtrace_ray_type"             ]->setUint( 0u );
    context[ "pathtrace_shadow_ray_type"      ]->setUint( 1u );
	context[ "pathtrace_insect_ray_type"      ]->setUint( 2u );
    context[ "rr_begin_depth"                 ]->setUint( rr_begin_depth );

    Buffer buffer = sutil::createOutputBuffer( context, RT_FORMAT_FLOAT4, width, height, use_pbo );
    context["output_buffer"]->set( buffer );

	Buffer tonemappedBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
	context["tonemapped_buffer"]->set(tonemappedBuffer);
	Buffer albedoBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);

	denoisedBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
	emptyBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, 0, 0);

    // Setup programs
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "optixPathTracer.cu" );
    context->setRayGenerationProgram( 0, context->createProgramFromPTXString( ptx, "pathtrace_camera" ) );
    context->setExceptionProgram( 0, context->createProgramFromPTXString( ptx, "exception" ) );
    context->setMissProgram( 0, context->createProgramFromPTXString( ptx, "miss" ) );
	context->setMissProgram( 1, context->createProgramFromPTXString(ptx, "insect_miss"));

    context[ "sqrt_num_samples" ]->setUint( sqrt_num_samples );
    context[ "bad_color"        ]->setFloat( 1000000.0f, 0.0f, 1000000.0f ); // Super magenta to make sure it doesn't get averaged out in the progressive rendering.
    context[ "bg_color"         ]->setFloat( make_float3(0.0f, 0.0f, 0.0f) );
}


void loadGeometry()
{
    // Light buffer
    ParallelogramLight light;
    light.corner   = make_float3( 343.0f, 548.6f, 227.0f);
    light.v1       = make_float3( -130.0f, 0.0f, 0.0f);
    light.v2       = make_float3( 0.0f, 0.0f, 105.0f);
    light.normal   = normalize( cross(light.v1, light.v2) );
    light.emission = make_float3( 30.0f, 30.0f, 30.0f );

    Buffer light_buffer = context->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( ParallelogramLight ) );
    light_buffer->setSize( 1u );
    memcpy( light_buffer->map(), &light, sizeof( light ) );
    light_buffer->unmap();
    context["lights"]->setBuffer( light_buffer );


	/* Load Mesh */
	std::string mesh_file = std::string(sutil::samplesDir()) + "/data/stanford_bunny.obj";
	OptiXMesh mesh;
	mesh.context = context;
	loadMesh(mesh_file, mesh, 
		Matrix4x4::translate(make_float3(278.0f, 0.0f, 279.6f)) * 
		Matrix4x4::rotate(MY_PI, make_float3(0.0f, 1.0f, 0.0f)) *
		//Matrix4x4::rotate(MY_PI*0.5f, make_float3(-1.0f, 0.0f, 0.0f)) *
		Matrix4x4::scale(make_float3(1500.0f)));

	/* Set Mesh Material */
	Material mesh_diffuse = context->createMaterial();
	const char *mesh_ptx = sutil::getPtxString(SAMPLE_NAME, "draw_color.cu");
	Program mdiffuse_ch = context->createProgramFromPTXString(mesh_ptx, "diffuse");
	Program mdiffuse_ah = context->createProgramFromPTXString(mesh_ptx, "shadow");
	Program mdiffuse_inse = context->createProgramFromPTXString(mesh_ptx, "find_insect");
	//Program mdiffuse_inse = context->createProgramFromPTXString(mesh_ptx, "find_insect_light");
	mesh_diffuse->setClosestHitProgram(0, mdiffuse_ch);
	mesh_diffuse->setAnyHitProgram(1, mdiffuse_ah);
	mesh_diffuse->setClosestHitProgram(2, mdiffuse_inse);

    // Set up material
    Material diffuse = context->createMaterial();
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "optixPathTracer.cu" );
    Program diffuse_ch = context->createProgramFromPTXString( ptx, "diffuse" );
    Program diffuse_ah = context->createProgramFromPTXString( ptx, "shadow" );
    diffuse->setClosestHitProgram( 0, diffuse_ch );
    diffuse->setAnyHitProgram( 1, diffuse_ah );
	//mesh_diffuse->setClosestHitProgram(2, mdiffuse_inse);

    Material diffuse_light = context->createMaterial();
    Program diffuse_em = context->createProgramFromPTXString( ptx, "diffuseEmitter" );
    diffuse_light->setClosestHitProgram( 0, diffuse_em );
	// Mention !!!!!

    // Set up parallelogram programs
    ptx = sutil::getPtxString( SAMPLE_NAME, "parallelogram.cu" );
    pgram_bounding_box = context->createProgramFromPTXString( ptx, "bounds" );
    pgram_intersection = context->createProgramFromPTXString( ptx, "intersect" );

    // create geometry instances
    std::vector<GeometryInstance> gis;

    const float3 white = make_float3( 0.8f, 0.8f, 0.8f );
    const float3 green = make_float3( 0.25f, 0.25f, 0.25f );
    const float3 red   = make_float3( 0.8f, 0.8f, 0.8f );
    const float3 light_em = make_float3( 130.0f, 130.0f, 130.0f );

	// Mesh
	GeometryInstance mesh_instance = mesh.geom_instance;
	mesh_instance->addMaterial(mesh_diffuse);	// !!!!!!!!!!!!!!!!!!!!!!!!!!  diffuse -> mesh_diffuse)
	mesh_instance["diffuse_color"]->setFloat(green);
	gis.push_back(mesh_instance);

	/*
	// Floor
	gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
		make_float3(0.0f, 0.0f, 559.2f),
		make_float3(556.0f, 0.0f, 0.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);

	// Ceiling
	gis.push_back(createParallelogram(make_float3(0.0f, 548.8f, 0.0f),
		make_float3(556.0f, 0.0f, 0.0f),
		make_float3(0.0f, 0.0f, 559.2f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);


	// Back wall
	gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 559.2f),
		make_float3(0.0f, 548.8f, 0.0f),
		make_float3(556.0f, 0.0f, 0.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);

	// Right wall
	gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
		make_float3(0.0f, 548.8f, 0.0f),
		make_float3(0.0f, 0.0f, 559.2f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);

	// Left wall
	gis.push_back(createParallelogram(make_float3(556.0f, 0.0f, 0.0f),
		make_float3(0.0f, 0.0f, 559.2f),
		make_float3(0.0f, 548.8f, 0.0f)));
	setMaterial(gis.back(), diffuse, "diffuse_color", white);
	*/
	// Short block
	/*gis.push_back( createParallelogram( make_float3( 130.0f, 165.0f, 65.0f),
	make_float3( -48.0f, 0.0f, 160.0f),
	make_float3( 160.0f, 0.0f, 49.0f) ) );
	setMaterial(gis.back(), mesh_diffuse, "diffuse_color", white);
	gis.push_back( createParallelogram( make_float3( 290.0f, 0.0f, 114.0f),
	make_float3( 0.0f, 165.0f, 0.0f),
	make_float3( -50.0f, 0.0f, 158.0f) ) );
	setMaterial(gis.back(), mesh_diffuse, "diffuse_color", white);
	gis.push_back( createParallelogram( make_float3( 130.0f, 0.0f, 65.0f),
	make_float3( 0.0f, 165.0f, 0.0f),
	make_float3( 160.0f, 0.0f, 49.0f) ) );
	setMaterial(gis.back(), mesh_diffuse, "diffuse_color", white);
	gis.push_back( createParallelogram( make_float3( 82.0f, 0.0f, 225.0f),
	make_float3( 0.0f, 165.0f, 0.0f),
	make_float3( 48.0f, 0.0f, -160.0f) ) );
	setMaterial(gis.back(), mesh_diffuse, "diffuse_color", white);
	gis.push_back( createParallelogram( make_float3( 240.0f, 0.0f, 272.0f),
	make_float3( 0.0f, 165.0f, 0.0f),
	make_float3( -158.0f, 0.0f, -47.0f) ) );
	setMaterial(gis.back(), mesh_diffuse, "diffuse_color", white);

	// Tall block
	gis.push_back( createParallelogram( make_float3( 423.0f, 330.0f, 247.0f),
	make_float3( -158.0f, 0.0f, 49.0f),
	make_float3( 49.0f, 0.0f, 159.0f) ) );
	setMaterial(gis.back(), mesh_diffuse, "diffuse_color", white);
	gis.push_back( createParallelogram( make_float3( 423.0f, 0.0f, 247.0f),
	make_float3( 0.0f, 330.0f, 0.0f),
	make_float3( 49.0f, 0.0f, 159.0f) ) );
	setMaterial(gis.back(), mesh_diffuse, "diffuse_color", white);
	gis.push_back( createParallelogram( make_float3( 472.0f, 0.0f, 406.0f),
	make_float3( 0.0f, 330.0f, 0.0f),
	make_float3( -158.0f, 0.0f, 50.0f) ) );
	setMaterial(gis.back(), mesh_diffuse, "diffuse_color", white);
	gis.push_back( createParallelogram( make_float3( 314.0f, 0.0f, 456.0f),
	make_float3( 0.0f, 330.0f, 0.0f),
	make_float3( -49.0f, 0.0f, -160.0f) ) );
	setMaterial(gis.back(), mesh_diffuse, "diffuse_color", white);
	gis.push_back( createParallelogram( make_float3( 265.0f, 0.0f, 296.0f),
	make_float3( 0.0f, 330.0f, 0.0f),
	make_float3( 158.0f, 0.0f, -49.0f) ) );
	setMaterial(gis.back(), mesh_diffuse, "diffuse_color", white);*/

    // Create shadow group (no light)
    GeometryGroup shadow_group = context->createGeometryGroup(gis.begin(), gis.end());
    shadow_group->setAcceleration( context->createAcceleration( "Trbvh" ) );
    context["top_shadower"]->set( shadow_group );

    // Light
    gis.push_back( createParallelogram( make_float3( 343.0f, 548.6f, 227.0f),
                                        make_float3( -130.0f, 0.0f, 0.0f),
                                        make_float3( 0.0f, 0.0f, 105.0f) ) );
    setMaterial(gis.back(), diffuse_light, "emission_color", light_em);

    // Create geometry group
    GeometryGroup geometry_group = context->createGeometryGroup(gis.begin(), gis.end());
    geometry_group->setAcceleration( context->createAcceleration( "Trbvh" ) );
    context["top_object"]->set( geometry_group );
}

  
void setupCamera()
{
    camera_eye    = make_float3( 278.0f, 273.0f, -900.0f );
    camera_lookat = make_float3( 278.0f, 273.0f,    0.0f );
    camera_up     = make_float3(   0.0f,   1.0f,    0.0f );

    camera_rotate  = Matrix4x4::identity();
}


void updateCamera()
{
    const float fov  = 35.0f;
    const float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
    
    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
            camera_u, camera_v, camera_w, /*fov_is_vertical*/ true );

    const Matrix4x4 frame = Matrix4x4::fromBasis( 
            normalize( camera_u ),
            normalize( camera_v ),
            normalize( -camera_w ),
            camera_lookat);
    const Matrix4x4 frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const Matrix4x4 trans     = frame*camera_rotate*camera_rotate*frame_inv; 

    camera_eye    = make_float3( trans*make_float4( camera_eye,    1.0f ) );
    camera_lookat = make_float3( trans*make_float4( camera_lookat, 1.0f ) );
    camera_up     = make_float3( trans*make_float4( camera_up,     0.0f ) );

    sutil::calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    camera_rotate = Matrix4x4::identity();

    if( camera_changed ) // reset accumulation
        frame_number = 1;
    camera_changed = false;

    context[ "frame_number" ]->setUint( frame_number++ );
    context[ "eye"]->setFloat( camera_eye );
    context[ "U"  ]->setFloat( camera_u );
    context[ "V"  ]->setFloat( camera_v );
    context[ "W"  ]->setFloat( camera_w );

}

// Post process #################################

bool           postprocessing_needs_init = true;

void setupPostprocessing()
{

	if (!tonemapStage)
	{
		// create stages only once: they will be reused in several command lists without being re-created
		tonemapStage = context->createBuiltinPostProcessingStage("TonemapperSimple");
		denoiserStage = context->createBuiltinPostProcessingStage("DLDenoiser");
		if (trainingDataBuffer)
		{
			Variable trainingBuff = denoiserStage->declareVariable("training_data_buffer");
			trainingBuff->set(trainingDataBuffer);
		}

		tonemapStage->declareVariable("input_buffer")->set(getOutputBuffer());
		tonemapStage->declareVariable("output_buffer")->set(context["tonemapped_buffer"]->getBuffer());
		tonemapStage->declareVariable("exposure")->setFloat(1.0f);
		tonemapStage->declareVariable("gamma")->setFloat(2.2f);

		//denoiserStage->declareVariable("input_buffer")->set(getOutputBuffer());
		denoiserStage->declareVariable("input_buffer")->set(context["tonemapped_buffer"]->getBuffer());
		denoiserStage->declareVariable("output_buffer")->set(denoisedBuffer);
		denoiserStage->declareVariable("blend")->setFloat(denoiseBlend);
		denoiserStage->declareVariable("input_albedo_buffer");
		denoiserStage->declareVariable("input_normal_buffer");
		
	}

	if (commandListWithDenoiser)
	{
		commandListWithDenoiser->destroy();
		commandListWithoutDenoiser->destroy();
	}


	commandListWithDenoiser = context->createCommandList();
	commandListWithDenoiser->appendLaunch(0, width, height);
	commandListWithDenoiser->appendPostprocessingStage(tonemapStage, width, height);
	commandListWithDenoiser->appendPostprocessingStage(denoiserStage, width, height);
	commandListWithDenoiser->finalize();

	postprocessing_needs_init = false;
}

// Post process finish ############

void glutInitialize( int* argc, char** argv )
{
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 1000, 1000 );                                               
    glutCreateWindow( SAMPLE_NAME );
    glutHideWindow();                                                              
}


void glutRun()
{
    // Initialize GL state                                                            
    glMatrixMode(GL_PROJECTION);                                                   
    glLoadIdentity();                                                              
    glOrtho(0, 1, 0, 1, -1, 1 );                                                   

    glMatrixMode(GL_MODELVIEW);                                                    
    glLoadIdentity();                                                              

    glViewport(0, 0, width, height);                                 

    glutShowWindow();                                                              
    glutReshapeWindow( width, height);

    // register glut callbacks
    glutDisplayFunc( glutDisplay );
    glutIdleFunc( glutDisplay );
    glutReshapeFunc( glutResize );
    glutKeyboardFunc( glutKeyboardPress );
    glutMouseFunc( glutMousePress );
    glutMotionFunc( glutMouseMotion );

    registerExitHandler();

    glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
    updateCamera();

	if (postprocessing_needs_init)
	{
		setupPostprocessing();
	}

	//Variable(denoiserStage->queryVariable("blend"))->setFloat(denoiseBlend);

	//commandListWithDenoiser->execute();

    context->launch( 0, width, height );

	//sutil::displayBufferGL(denoisedBuffer, BUFFER_PIXEL_FORMAT_DEFAULT, true);
	sutil::displayBufferGL( getOutputBuffer() );

    {
      static unsigned frame_count = 0;
      sutil::displayFps( frame_count++ );
    }

    glutSwapBuffers();
}


void glutKeyboardPress( unsigned char k, int x, int y )
{

    switch( k )
    {
        case( 'q' ):
        case( 27 ): // ESC
        {
            destroyContext();
            exit(0);
        }
        case( 's' ):
        {
            const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
            std::cerr << "Saving current frame to '" << outputImage << "'\n";
            sutil::displayBufferPPM( outputImage.c_str(), getOutputBuffer(), false );
            break;
        }
    }
}


void glutMousePress( int button, int state, int x, int y )
{
    if( state == GLUT_DOWN )
    {
        mouse_button = button;
        mouse_prev_pos = make_int2( x, y );
    }
    else
    {
        // nothing
    }
}


void glutMouseMotion( int x, int y)
{
    if( mouse_button == GLUT_RIGHT_BUTTON )
    {
        const float dx = static_cast<float>( x - mouse_prev_pos.x ) /
                         static_cast<float>( width );
        const float dy = static_cast<float>( y - mouse_prev_pos.y ) /
                         static_cast<float>( height );
        const float dmax = fabsf( dx ) > fabs( dy ) ? dx : dy;
        const float scale = std::min<float>( dmax, 0.9f );
        camera_eye = camera_eye + (camera_lookat - camera_eye)*scale;
        camera_changed = true;
    }
    else if( mouse_button == GLUT_LEFT_BUTTON )
    {
        const float2 from = { static_cast<float>(mouse_prev_pos.x),
                              static_cast<float>(mouse_prev_pos.y) };
        const float2 to   = { static_cast<float>(x),
                              static_cast<float>(y) };

        const float2 a = { from.x / width, from.y / height };
        const float2 b = { to.x   / width, to.y   / height };

        camera_rotate = arcball.rotate( b, a );
        camera_changed = true;
    }

    mouse_prev_pos = make_int2( x, y );
}


void glutResize( int w, int h )
{
    if ( w == (int)width && h == (int)height ) return;

    camera_changed = true;

    width  = w;
    height = h;
    
    sutil::resizeBuffer( getOutputBuffer(), width, height );
	sutil::resizeBuffer(context["tonemapped_buffer"]->getBuffer(), width, height);
	sutil::resizeBuffer(context["input_albedo_buffer"]->getBuffer(), width, height);
	sutil::resizeBuffer(context["input_normal_buffer"]->getBuffer(), width, height);
	sutil::resizeBuffer(denoisedBuffer, width, height);

    glViewport(0, 0, width, height);                                               

    glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
        "App Options:\n"
        "  -h | --help               Print this usage message and exit.\n"
        "  -f | --file               Save single frame to file and exit.\n"
        "  -n | --nopbo              Disable GL interop for display buffer.\n"
        "  -m | --mesh <mesh_file>   Specify path to mesh to be loaded.\n"
        "App Keystrokes:\n"
        "  q  Quit\n" 
        "  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
        << std::endl;

    exit(1);
}


int main( int argc, char** argv )
 {
    std::string out_file;
    std::string mesh_file = std::string( sutil::samplesDir() ) + "/data/cow.obj";
    for( int i=1; i<argc; ++i )
    {
        const std::string arg( argv[i] );

        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "-f" || arg == "--file"  )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            out_file = argv[++i];
        }
        else if( arg == "-n" || arg == "--nopbo"  )
        {
            use_pbo = false;
        }
        else if( arg == "-m" || arg == "--mesh" )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << argv[i] << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            mesh_file = argv[++i];
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        glutInitialize( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif

        createContext();
        setupCamera();
        loadGeometry();

        context->validate();

        /*if ( out_file.empty() )
        {
            glutRun();
        }
        else
        {*/
            updateCamera();
            context->launch( 0, width, height );
            sutil::displayBufferPPM( "./result.ppm", getOutputBuffer(), false );
            destroyContext();
        //}

        return 0;
    }
	SUTIL_CATCH(context->get())
	while (1);
}

