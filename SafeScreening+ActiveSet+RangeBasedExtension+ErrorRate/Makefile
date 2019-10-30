###################################################
# Makefile (*.cpp���R���p�C������ꍇ)
# �R���p�C���������\�[�X������t�H���_�Ɠ����Ƃ���ɒu��
# �f�B���N�g���K�w�}
# this
# |--- *.cpp          (�R���p�C���������\�[�X�t�@�C��)
# |--- Makefile
# |--- run.exe        (�����������s�t�@�C��)
# |--- obj            (���������t�H���_)
#      |--- *.o       (���������I�u�W�F�N�g�t�@�C��)
#      |--- *.d       (���������ˑ��֌W�t�@�C��)
#
# ���s�R�}���h
# make         (�X�V���ꂽ�t�@�C���̂݃R���p�C��)
# make all     (�N���[�����ăr���h)
# make clean   (�N���[��)
###################################################
AR       = ar
ARFLAGS  = rus

CXX      = g++
CXXFLAGS = -std=c++11 -MMD -MP -Wall -Wextra -Winit-self -Wno-unused-parameter -DEIGEN_NO_DEBUG -O3 -fopenmp

RM       = rm -f
LDFLAGS  = -fopenmp
LIBS     =
INCLUDE  =  -I../../../C++/Eigen
#���s�t�@�C��
TARGET   = ./run
#�I�u�W�F�N�g�t�@�C���ۑ��t�H���_
OBJDIR   = ./obj
#�\�[�X�t�@�C��
SOURCES  = $(wildcard *.cpp)
#�I�u�W�F�N�g�t�@�C��
OBJECTS  = $(addprefix $(OBJDIR)/, $(SOURCES:.cpp=.o))
#�ˑ��֌W�t�@�C��
DEPENDS  = $(OBJECTS:.o=.d)
#���s�t�@�C���쐬
$(TARGET): $(OBJECTS) $(LIBS)
	$(CXX) -o $@ $^ $(LDFLAGS)
#�\�[�X�t�@�C�����R���p�C��
$(OBJDIR)/%.o: %.cpp
	@if [ ! -d $(OBJDIR) ];\
	then echo "mkdir -p $(OBJDIR)";mkdir -p $(OBJDIR);\
	fi
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<

#�N���[�����ăr���h
.PHONY: all
all: clean $(TARGET)

#�N���[��
.PHONY:clean
clean:
	$(RM) $(OBJECTS) $(DEPENDS) $(TARGET)

-include $(DEPENDS)