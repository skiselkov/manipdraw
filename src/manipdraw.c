/*
 * Copyright 2023 Saso Kiselkov. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * “Software”), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
 * NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <stdbool.h>
#include <time.h>

#include <XPLMDisplay.h>
#include <XPLMPlugin.h>

#include <cglm/cglm.h>

#include <acfutils/crc64.h>
#include <acfutils/dr.h>
#include <acfutils/glew.h>
#include <acfutils/log.h>
#include <acfutils/helpers.h>
#include <acfutils/osrand.h>
#include <acfutils/shader.h>

#include <obj8.h>

#define	PLUGIN_NAME		"manipdraw"
#define	PLUGIN_SIG		"skiselkov.manipdraw"
#define	PLUGIN_DESCRIPTION	"manipdraw"

static struct {
	dr_t	fbo;
	dr_t	viewport;
	dr_t	acf_matrix;
	dr_t	mv_matrix;
	dr_t	proj_matrix_3d;
	dr_t	rev_float_z;
	dr_t	modern_drv;
} drs;

static int		xpver = 0;
static char		plugindir[512] = { 0 };

static GLuint		cursor_tex[2] = {};
static GLuint		cursor_fbo = 0;
static GLuint		cursor_pbo = 0;
static bool		cursor_xfer = false;
static GLushort		manip_idx = UINT16_MAX;

static uint64_t		last_draw_t = 0;
static uint64_t		blink_start_t = 0;
static GLushort		prev_manip_idx = UINT16_MAX;

static shader_info_t generic_vert_info = { .filename = "generic.vert.spv" };
static shader_info_t resolve_frag_info = { .filename = "resolve.frag.spv" };
static shader_info_t paint_frag_info = { .filename = "paint.frag.spv" };
static const shader_prog_info_t resolve_prog_info = {
    .progname = "manipdraw_resolve",
    .vert = &generic_vert_info,
    .frag = &resolve_frag_info
};
static const shader_prog_info_t paint_prog_info = {
    .progname = "manipdraw_paint",
    .vert = &generic_vert_info,
    .frag = &paint_frag_info
};
static shader_obj_t	resolve_shader = {};
static shader_obj_t	paint_shader = {};
static obj8_t		*obj = NULL;

enum {
    U_PVM,
    U_ALPHA,
    NUM_UNIFORMS
};
static const char *uniforms[NUM_UNIFORMS] = {
    [U_PVM] = "pvm",
    [U_ALPHA] = "alpha"
};

static void
resolve_manip_complete(void)
{
	GLushort *data;

	/* No transfer in progress, so allow caller to start a new update */
	if (!cursor_xfer)
		return;

	ASSERT(cursor_pbo != 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, cursor_pbo);
	data = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
	if (data != NULL) {
		/* single pixel containing the clickspot index */
		manip_idx = *data;
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
	}
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	cursor_xfer = false;
}

static bool
is_rev_float_z(void)
{
	return (xpver >= 12000 || dr_geti(&drs.modern_drv) != 0 ||
	    dr_geti(&drs.rev_float_z) != 0);
}

static void
resolve_manip(int mouse_x, int mouse_y, const mat4 pvm)
{
	int vp[4];

	ASSERT(pvm != NULL);

	resolve_manip_complete();

	VERIFY3S(dr_getvi(&drs.viewport, vp, 0, 4), ==, 4);

	ASSERT(cursor_fbo != 0);
	glBindFramebufferEXT(GL_FRAMEBUFFER, cursor_fbo);
	glViewport(vp[0] - mouse_x, vp[1] - mouse_y, vp[2], vp[3]);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	if (is_rev_float_z()) {
		glDepthFunc(GL_GREATER);
		glClearDepth(0);
	}
	/*
	 * We want to set the FBO's color to 1, which is 0xFFFF in 16-bit.
	 * That way, if nothing covers it, we know that there is no valid
	 * manipulator there.
	 */
	glClearColor(1, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0, 0, 0, 0);

	shader_obj_bind(&resolve_shader);
	glUniformMatrix4fv(shader_obj_get_u(&resolve_shader, U_PVM),
	    1, GL_FALSE, (const GLfloat *)pvm);
	ASSERT(obj != NULL);
	obj8_set_render_mode(obj, OBJ8_RENDER_MODE_MANIP_ONLY);
	obj8_draw_group(obj, NULL, shader_obj_get_prog(&resolve_shader), pvm);

	ASSERT(cursor_pbo != 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, cursor_pbo);
	glReadPixels(0, 0, 1, 1, GL_RED, GL_UNSIGNED_SHORT, NULL);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	cursor_xfer = true;
	/*
	 * Restore original XP viewport & framebuffer binding.
	 */
	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);
	if (is_rev_float_z()) {
		glDepthFunc(GL_LESS);
		glClearDepth(1);
	}
	glBindFramebufferEXT(GL_FRAMEBUFFER, dr_geti(&drs.fbo));
	glViewport(vp[0], vp[1], vp[2], vp[3]);
}

static void
paint_manip(const mat4 pvm)
{
	uint64_t now = microclock(), delta_t = 0;
	int vp[4];
	float alpha;

	ASSERT(pvm != NULL);

	VERIFY3S(dr_getvi(&drs.viewport, vp, 0, 4), ==, 4);

	if (manip_idx != prev_manip_idx || now - last_draw_t > SEC2USEC(0.2)) {
		blink_start_t = now;
		prev_manip_idx = manip_idx;
	}
	last_draw_t = now;
	delta_t = (now - blink_start_t) % 1000000;
	if (delta_t < 500000)
		alpha = delta_t / 500000.0;
	else
		alpha = 1 - (delta_t - 500000) / 500000.0;

	shader_obj_bind(&paint_shader);
	glUniformMatrix4fv(shader_obj_get_u(&paint_shader, U_PVM),
	    1, GL_FALSE, (const GLfloat *)pvm);
	glUniform1f(shader_obj_get_u(&paint_shader, U_ALPHA), alpha);
	ASSERT(obj != NULL);
	glEnable(GL_BLEND);
	obj8_set_render_mode2(obj, OBJ8_RENDER_MODE_MANIP_ONLY_ONE, manip_idx);
	obj8_draw_group(obj, NULL, shader_obj_get_prog(&paint_shader), pvm);

	glViewport(vp[0], vp[1], vp[2], vp[3]);
}

static int
draw_cb(XPLMDrawingPhase phase, int before, void *refcon)
{
	int mouse_x, mouse_y;
	int vp[4];
	mat4 proj_matrix, acf_matrix, pvm;

	UNUSED(phase);
	UNUSED(before);
	UNUSED(refcon);

	XPLMGetMouseLocationGlobal(&mouse_x, &mouse_y);
	VERIFY3S(dr_getvi(&drs.viewport, vp, 0, 4), ==, 4);

	if (mouse_x < vp[0] || mouse_x > vp[0] + vp[2] ||
	    mouse_y < vp[1] || mouse_y > vp[1] + vp[3]) {
		/* Mouse off-screen, don't draw anything */
		return (1);
	}
	/*
	 * Mouse is somewhere on the screen. Redraw the manipulator stack.
	 */
	shader_obj_reload_check(&resolve_shader);
	shader_obj_reload_check(&paint_shader);

	dr_getvf32(&drs.acf_matrix, (float *)acf_matrix, 0, 16);
	dr_getvf32(&drs.proj_matrix_3d, (float *)proj_matrix, 0, 16);
	glm_mat4_mul(proj_matrix, acf_matrix, pvm);

	UNUSED(resolve_manip);
	resolve_manip(mouse_x, mouse_y, pvm);
	if (manip_idx != UINT16_MAX)
		paint_manip(pvm);
	glUseProgram(0);

	return (1);
}

static void
create_cursor_objects(void)
{
	/*
	 * Create the textures which will hold the rendered manipulator
	 * pixel right under the user's cursor spot. We need two textures
	 * here, one to hold the manipulator ID (16-bit single-channel
	 * texture, using the GL_RED channel), and another one to hold
	 * the depth buffer (to properly handle depth and occlusion).
	 */
	glGenTextures(ARRAY_NUM_ELEM(cursor_tex), cursor_tex);
	VERIFY(cursor_tex[0] != 0);
	setup_texture(cursor_tex[0], GL_R16, 1, 1,
	    GL_RED, GL_UNSIGNED_SHORT, NULL);
	setup_texture(cursor_tex[1], GL_DEPTH_COMPONENT32F, 1, 1,
	    GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	/*
	 * Set up the framebuffer object. This will be the target to draw
	 * the manipulator IDs. The contents of the framebuffer will be
	 * backed by the textures created above.
	 */
	glGenFramebuffers(1, &cursor_fbo);
	VERIFY(cursor_fbo != 0);
	setup_color_fbo_for_tex(cursor_fbo, cursor_tex[0], cursor_tex[1], 0,
	    false);
	/*
	 * Set up the back-transfer pixel buffer. This is used to retrieve
	 * the manipulator render result back from GPU VRAM.
	 */
	glGenBuffers(1, &cursor_pbo);
	VERIFY(cursor_pbo != 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, cursor_pbo);
	glBufferData(GL_PIXEL_PACK_BUFFER, sizeof (GLushort), NULL,
	    GL_STREAM_READ);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

static void
destroy_cursor_objects(void)
{
	if (cursor_pbo != 0) {
		glDeleteBuffers(1, &cursor_pbo);
		cursor_pbo = 0;
	}
	if (cursor_fbo != 0) {
		glDeleteFramebuffers(1, &cursor_fbo);
		cursor_fbo = 0;
	}
	if (cursor_tex[0] != 0) {
		glDeleteTextures(ARRAY_NUM_ELEM(cursor_tex), cursor_tex);
		memset(cursor_tex, 0, sizeof (cursor_tex));
	}
}

static void
log_dbg_string(const char *str)
{
	XPLMDebugString(str);
}

PLUGIN_API int
XPluginStart(char *name, char *sig, char *desc)
{
	char *p;
	GLenum err;
	uint64_t seed;
	int xplm_ver;
	XPLMHostApplicationID host_id;
	/*
	 * libacfutils logging facility bootstrap, this must be one of
	 * the first steps during init, to make sure we have the logMsg
	 * and general error logging facilities available early.
	 */
	log_init(log_dbg_string, "manipdraw");

	ASSERT(name != NULL);
	ASSERT(sig != NULL);
	ASSERT(desc != NULL);
	XPLMGetVersions(&xpver, &xplm_ver, &host_id);
	/*
	 * Always use Unix-native paths on the Mac!
	 */
	XPLMEnableFeature("XPLM_USE_NATIVE_PATHS", 1);
	XPLMEnableFeature("XPLM_USE_NATIVE_WIDGET_WINDOWS", 1);
	/*
	 * Construct plugindir to point to our plugin's root directory.
	 */
	XPLMGetPluginInfo(XPLMGetMyID(), NULL, plugindir, NULL, NULL);
	fix_pathsep(plugindir);
	/* cut off the trailing path component (our filename) */
	if ((p = strrchr(plugindir, DIRSEP)) != NULL)
		*p = '\0';
	/* cut off an optional '32' or '64' trailing component */
	if ((p = strrchr(plugindir, DIRSEP)) != NULL) {
		if (strcmp(p + 1, "64") == 0 || strcmp(p + 1, "32") == 0 ||
		    strcmp(p + 1, "win_x64") == 0 ||
		    strcmp(p + 1, "mac_x64") == 0 ||
		    strcmp(p + 1, "lin_x64") == 0)
			*p = '\0';
	}
	/*
	 * Initialize the CRC64 and PRNG machinery inside of libacfutils.
	 */
	crc64_init();
	if (!osrand(&seed, sizeof (seed)))
		seed = microclock() + clock();
	crc64_srand(seed);
	/*
	 * GLEW bootstrap
	 */
	err = glewInit();
	if (err != GLEW_OK) {
		/* Problem: glewInit failed, something is seriously wrong. */
		logMsg("FATAL ERROR: cannot initialize libGLEW: %s",
		    glewGetErrorString(err));
		goto errout;
	}
	if (!GLEW_VERSION_2_1) {
		logMsg("FATAL ERROR: your system doesn't support OpenGL 2.1");
		goto errout;
	}
	strcpy(name, PLUGIN_NAME);
	strcpy(sig, PLUGIN_SIG);
	strcpy(desc, PLUGIN_DESCRIPTION);

	return (1);
errout:
	return (0);
}

PLUGIN_API void
XPluginStop(void)
{
	log_fini();
}

PLUGIN_API int
XPluginEnable(void)
{
	char *shader_dir, *obj_path;

	fdr_find(&drs.fbo, "sim/graphics/view/current_gl_fbo");
	fdr_find(&drs.viewport, "sim/graphics/view/viewport");
	fdr_find(&drs.acf_matrix, "sim/graphics/view/acf_matrix");
	fdr_find(&drs.mv_matrix, "sim/graphics/view/modelview_matrix");
	fdr_find(&drs.proj_matrix_3d, "sim/graphics/view/projection_matrix_3d");
	if (!dr_find(&drs.rev_float_z,
	    "sim/graphics/view/is_reverse_float_z") ||
	    !dr_find(&drs.modern_drv,
	    "sim/graphics/view/using_modern_driver")) {
		ASSERT3S(xpver, >=, 12000);
	}
	VERIFY(XPLMRegisterDrawCallback(draw_cb, xplm_Phase_Window, 1, NULL));

	create_cursor_objects();

	shader_dir = mkpathname(plugindir, "shaders", NULL);
	if (!shader_obj_init(&resolve_shader, shader_dir, &resolve_prog_info,
	    NULL, 0, uniforms, NUM_UNIFORMS) ||
	    !shader_obj_init(&paint_shader, shader_dir, &paint_prog_info,
	    NULL, 0, uniforms, NUM_UNIFORMS)) {
		goto errout;
	}
	obj_path = mkpathname(plugindir, "..", "..", "objects",
	    "CL650_cockpit.obj", NULL);
	obj = obj8_parse(obj_path, ZERO_VECT3);
	lacf_free(obj_path);
	if (obj == NULL)
		goto errout;

	lacf_free(shader_dir);
	return (1);
errout:
	lacf_free(shader_dir);
	return (0);
}

PLUGIN_API void
XPluginDisable(void)
{
	XPLMUnregisterDrawCallback(draw_cb, xplm_Phase_Window, 1, NULL);

	destroy_cursor_objects();
	shader_obj_fini(&resolve_shader);
	shader_obj_fini(&paint_shader);
	if (obj != NULL) {
		obj8_free(obj);
		obj = NULL;
	}
}

PLUGIN_API void
XPluginReceiveMessage(XPLMPluginID from, int msg, void *param)
{
	UNUSED(from);
	UNUSED(msg);
	UNUSED(param);
}
