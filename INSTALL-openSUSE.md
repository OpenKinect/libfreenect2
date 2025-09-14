# Installation de libfreenect2 sur openSUSE

Ce guide explique comment installer libfreenect2 (driver pour Kinect v2) sur openSUSE.

## Prérequis

- openSUSE Leap 15.4+ ou Tumbleweed
- Port USB 3.0 (le Kinect v2 ne fonctionne pas avec USB 2.0)
- Contrôleur USB 3.0 compatible (Intel et NEC sont connus pour fonctionner, ASMedia ne fonctionne pas)

## Installation automatique

Le moyen le plus simple est d'utiliser le script d'installation fourni :

```bash
cd /chemin/vers/libfreenect2
./install_opensuse.sh
```

## Installation manuelle

### 1. Installer les outils de base

```bash
sudo zypper install git cmake pkg-config gcc-c++ make
```

### 2. Installer les dépendances principales

```bash
# libusb (requis, version >= 1.0.20)
sudo zypper install libusb-1_0-devel

# TurboJPEG pour le décodage JPEG rapide
sudo zypper install libjpeg-turbo libturbojpeg0

# GLFW pour OpenGL
sudo zypper install libglfw3 libglfw-devel

# OpenGL/Mesa
sudo zypper install Mesa-devel Mesa-libGL-devel Mesa-libEGL-devel
```

### 3. Dépendances optionnelles

#### OpenCL (pour accélération GPU)

```bash
# Headers OpenCL
sudo zypper install opencl-headers

# Pour GPU Intel
sudo zypper install intel-compute-runtime

# Pour GPU AMD, installer les drivers propriétaires AMD
```

#### CUDA (pour GPU NVIDIA)

```bash
# Suivre les instructions NVIDIA pour installer CUDA Toolkit
# Généralement via les dépôts NVIDIA ou les packages téléchargés
```

#### VAAPI (décodage JPEG Intel)

```bash
sudo zypper install libva-devel
```

#### OpenNI2 (optionnel)

```bash
sudo zypper install libopenni2-devel
```

### 4. Compiler libfreenect2

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/freenect2
make -j$(nproc)
make install
```

### 5. Configuration des permissions

```bash
sudo cp ../platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
# Débrancher et rebrancher le Kinect
```

## Test de l'installation

```bash
cd $HOME/freenect2/bin
./Protonect
```

## Différences avec Ubuntu/Debian

| Ubuntu/Debian | openSUSE | Notes |
|---------------|----------|-------|
| `apt-get install` | `zypper install` | Gestionnaire de paquets |
| `libturbojpeg0-dev` | `libjpeg-turbo` + `libturbojpeg0` | Paquets séparés |
| `libglfw3-dev` | `libglfw3` + `libglfw-devel` | Séparation lib/devel |
| `beignet-dev` | `intel-compute-runtime` | OpenCL Intel |

## Résolution de problèmes

### Le Kinect n'est pas détecté

1. Vérifiez que c'est bien un port USB 3.0 : `lsusb -t`
2. Vérifiez les règles udev : `ls -la /etc/udev/rules.d/*kinect*`
3. Redémarrez le service udev : `sudo systemctl restart systemd-udevd`
4. Vérifiez les logs : `dmesg | grep -i kinect`

### Erreurs de compilation

1. Vérifiez que toutes les dépendances sont installées
2. Pour les erreurs OpenCL : `cmake .. -DENABLE_OPENCL=OFF`
3. Pour les erreurs CUDA : `cmake .. -DENABLE_CUDA=OFF`
4. Pour les erreurs OpenGL : `cmake .. -DENABLE_OPENGL=OFF`

### Performance

- Utilisez différents pipelines via la variable d'environnement :
  ```bash
  export LIBFREENECT2_PIPELINE=opengl  # (défaut)
  export LIBFREENECT2_PIPELINE=cl      # OpenCL
  export LIBFREENECT2_PIPELINE=cuda    # CUDA
  export LIBFREENECT2_PIPELINE=cpu     # CPU seulement (lent)
  export LIBFREENECT2_PIPELINE=vaapi   # VAAPI (Intel)
  ```

## Intégration avec des projets tiers

Pour utiliser libfreenect2 dans vos propres projets CMake :

```cmake
find_package(freenect2 REQUIRED 
  HINTS $ENV{HOME}/freenect2/lib/cmake/freenect2)
target_link_libraries(your_target ${freenect2_LIBRARIES})
```

## Désinstallation

```bash
# Supprimer l'installation locale
rm -rf $HOME/freenect2

# Supprimer les règles udev
sudo rm /etc/udev/rules.d/90-kinect2.rules
sudo udevadm control --reload-rules
```
