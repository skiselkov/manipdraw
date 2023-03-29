# Copyright 2023 Saso Kiselkov. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# “Software”), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

.PHONY : all clean

SPVS = \
    generic.vert.spv \
    resolve.frag.spv \
    paint.frag.spv

OUTDIR=..
SPIRVX_TGT_VERSION=120
SPIRVX_420_VERSION=420
SPIRVX_430_VERSION=430
SPIRVX_450_VERSION=450
SPIRVX_460_VERSION=460

GLSLANG=glslangValidator
SPIRVX=spirv-cross
GLSL_CLEANUP=./glsl_cleanup.sh
ECHO=/bin/echo

ifeq ($(V),1)
	VERB=
define logMsg
endef
else	# Not Verbose
	VERB=@
define logMsg
	@$(ECHO) $(1)
endef
endif

define BUILD_SHADER
	$(call logMsg,-n \	[GLSLANG]\	)
	$(VERB) $(GLSLANG) $(2) -G -o $@ $^

	$(call logMsg,\	[SPIRVX]\	$@)
	$(VERB) $(SPIRVX) --version $(SPIRVX_TGT_VERSION) \
	    --extension GL_EXT_gpu_shader4 \
	    --output $(@:%.$(1).spv=%.$(1).glsl) $@
	$(VERB) $(GLSL_CLEANUP) $(@:%.$(1).spv=%.$(1).glsl)

	$(call logMsg,\	[SPIRVX 4.20]\	$(@:%.$(1).spv=%.$(1).glsl420))
	$(VERB) $(SPIRVX) --version $(SPIRVX_420_VERSION) \
	    --output $(@:%.$(1).spv=%.$(1).glsl420) $@
	$(VERB) $(GLSL_CLEANUP) $(@:%.$(1).spv=%.$(1).glsl420)

	$(call logMsg,\	[SPIRVX 4.60]\	$(@:%.$(1).spv=%.$(1).glsl460))
	$(VERB) $(SPIRVX) --version $(SPIRVX_460_VERSION) \
	    --output $(@:%.$(1).spv=%.$(1).glsl460) $@
	$(VERB) $(GLSL_CLEANUP) $(@:%.$(1).spv=%.$(1).glsl460)
endef

SPVS_OUT=$(addprefix $(OUTDIR)/,$(SPVS))
all : $(SPVS_OUT)

clean :
	rm -f $(SPVS_OUT) $(patsubst %.spv,%.glsl,$(SPVS_OUT))

$(OUTDIR)/%.vert.spv : %.vert
	$(call BUILD_SHADER,vert)

$(OUTDIR)/%.frag.spv : %.frag
	$(call BUILD_SHADER,frag)

$(addprefix $(OUTDIR)/,$(SPVS)) : | $(OUTDIR)
