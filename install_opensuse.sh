#!/bin/bash

# Script d'installation de libfreenect2 pour openSUSE
# Adapté du script d'installation Ubuntu/Debian

set -e

echo "=== Installation de libfreenect2 sur openSUSE ==="
echo ""

# Fonction pour vérifier si une commande existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Vérifier si on a les privilèges admin
if ! command_exists sudo && [ "$EUID" -ne 0 ]; then
    echo "Erreur: Ce script nécessite les privilèges administrateur."
    echo "Utilisez 'sudo' ou exécutez en tant que root."
    exit 1
fi

# Fonction pour installer avec zypper
install_pkg() {
    if command_exists sudo; then
        sudo zypper install -y "$@"
    else
        zypper install -y "$@"
    fi
}

echo "1. Installation des outils de compilation de base..."
install_pkg git cmake pkg-config gcc-c++ make

echo ""
echo "2. Installation des dépendances libfreenect2..."

# libusb >= 1.0.20 (requis)
echo "   - libusb-1.0 (développement)"
install_pkg libusb-1_0-devel

# TurboJPEG pour le décodage JPEG
echo "   - TurboJPEG"
install_pkg libjpeg-turbo libturbojpeg0-devel

# GLFW pour OpenGL
echo "   - GLFW3"
install_pkg glfw-devel

# OpenGL (Mesa)
echo "   - OpenGL/Mesa"
install_pkg Mesa-devel Mesa-libGL1 Mesa-libEGL1 Mesa-libGLU1

echo ""
echo "3. Installation des dépendances optionnelles..."

# OpenCL (optionnel, pour Intel/AMD)
echo "   - OpenCL (optionnel)"
install_pkg opencl-headers || echo "     OpenCL headers non trouvés, continuons..."

# Pour Intel GPU OpenCL
echo "   - Intel OpenCL (si GPU Intel présent)"
install_pkg intel-compute-runtime || echo "     Intel OpenCL runtime non trouvé, continuons..."

# VAAPI pour décodage JPEG Intel (optionnel)
echo "   - VAAPI (optionnel, Intel seulement)"
install_pkg libva-devel || echo "     VAAPI non trouvé, continuons..."

# OpenNI2 (optionnel)
echo "   - OpenNI2 (optionnel)"
install_pkg libopenni2-devel || echo "     OpenNI2 non trouvé, continuons..."

echo ""
echo "4. Configuration des permissions USB..."
# Copier les règles udev
if command_exists sudo; then
    sudo cp platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules
else
    cp platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/
    udevadm control --reload-rules
fi

echo ""
echo "5. Compilation de libfreenect2..."

# Créer le répertoire de build s'il n'existe pas
mkdir -p build
cd build

# Configuration avec CMake
echo "   Configuration avec CMake..."
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/freenect2 \
         -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Compilation
echo "   Compilation..."
make -j$(nproc)

# Installation
echo "   Installation..."
make install

echo ""
echo "=== Installation terminée avec succès ! ==="
echo ""
echo "Pour tester l'installation :"
echo "1. Branchez votre Kinect v2"
echo "2. Exécutez : cd $HOME/freenect2/bin && ./Protonect"
echo ""
echo "Notes importantes :"
echo "- Vous devez débrancher/rebrancher votre Kinect après l'installation des règles udev"
echo "- Le Kinect v2 nécessite un port USB 3.0"
echo "- Pour les applications tiers utilisant libfreenect2, utilisez :"
echo "  cmake -Dfreenect2_DIR=$HOME/freenect2/lib/cmake/freenect2"
echo ""
echo "Pour les fonctionnalités optionnelles :"
echo "- OpenCL : définissez LIBFREENECT2_PIPELINE=cl"
echo "- CUDA : définissez LIBFREENECT2_PIPELINE=cuda (nécessite installation séparée)"
echo "- VAAPI : définissez LIBFREENECT2_PIPELINE=vaapi (Intel seulement)"
