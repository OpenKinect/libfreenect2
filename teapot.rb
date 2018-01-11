
#
#  This file is part of the "Teapot" project, and is released under the MIT license.
#

teapot_version "1.0.0"

define_target "libfreenect2" do |target|
	target.build do
		source_files = target.package.path
		cache_prefix = Files::Directory.join(environment[:build_prefix], "libfreenect2-#{environment.checksum}")
		package_files = Path.join(environment[:install_prefix], "lib/pkgconfig/freenect2.pc")
		
		cmake source: source_files, build_prefix: cache_prefix, arguments: [
			"-DBUILD_SHARED_LIBS=OFF",
		]
		
		make prefix: cache_prefix, package_files: package_files
	end
	
	target.depends :platform
	
	target.depends "Build/Make"
	target.depends "Build/CMake"
	
	target.provides "Library/freenect2" do
		append linkflags "-lfreenect2"
	end
end
