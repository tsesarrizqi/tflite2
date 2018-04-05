Build:
bazel build -c opt --cxxopt='--std=c++11' --linkopt='-llog' --linkopt='-lOpenCL' //tensorflow/contrib/lite/java:tensorflowlite --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=armeabi-v7a

bazel build -c opt --cxxopt='--std=c++11' --linkopt='-llog' --linkopt='-lOpenCL' --linkopt='-lvulkan' --linkopt='-lshaderc' --linkopt='-lgnustl_static' --linkopt='-lVkLayer_core_validation' --linkopt='-lVkLayer_threading' --linkopt='-lVkLayer_parameter_validation' --linkopt='-lVkLayer_object_tracker' --linkopt='-lVkLayer_image' --linkopt='-lVkLayer_swapchain' --linkopt='-lVkLayer_unique_objects' //tensorflow/contrib/lite/java:tensorflowlite --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=armeabi-v7a

