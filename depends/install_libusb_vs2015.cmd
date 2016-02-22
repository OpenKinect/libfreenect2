rem This can only be run in a Git Shell or similar environments
rem with access to git.exe and msbuild.exe.

rmdir /s /q libusb_src libusb

git clone https://github.com/libusb/libusb.git libusb_src || exit /b
cd libusb_src

git remote add xlz https://github.com/xlz/libusb.git
git fetch xlz || exit /b
git merge xlz/winiso

set CONFIG=Release
msbuild msvc\libusb_dll_2015.vcxproj /p:Platform=x64 /p:Configuration=%CONFIG% /target:Rebuild

mkdir ..\libusb\include\libusb-1.0
copy libusb\libusb.h ..\libusb\include\libusb-1.0
mkdir ..\libusb\MS64\dll
copy x64\%CONFIG%\dll\libusb-1.0.lib ..\libusb\MS64\dll
copy x64\%CONFIG%\dll\libusb-1.0.dll ..\libusb\MS64\dll
copy x64\%CONFIG%\dll\libusb-1.0.pdb ..\libusb\MS64\dll
