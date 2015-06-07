#!/bin/sh

cd `dirname $0`
DEPENDS_DIR=`pwd`

# download standalone packages for 14.04 LTS
wget http://launchpadlibrarian.net/173940430/libglfw3_3.0.4-1_amd64.deb
wget http://launchpadlibrarian.net/173940431/libglfw3-dev_3.0.4-1_amd64.deb
wget http://launchpadlibrarian.net/173940397/libglfw3-doc_3.0.4-1_all.deb

sudo dpkg -i libglfw3*_3.0.4-1_amd64.deb

sh ./install_deps.sh
