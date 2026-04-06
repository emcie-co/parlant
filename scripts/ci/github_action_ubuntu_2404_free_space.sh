#!/bin/sh

# Print initial disk space usage
df -h / | awk 'NR==2 {printf "Before cleanup: %s used, %s free\n", $3, $4}'

# Remove docker images
sudo docker rmi $(docker image ls -aq) >/dev/null 2>&1 || true

# Remove development toolchains and SDK directories
sudo rm -rf \
  /opt/hostedtoolcache/* \
  /usr/local/lib/android \
  /usr/share/dotnet \
  /usr/local/share/powershell \
  /usr/share/swift \
  /opt/ghc \
  /usr/local/.ghcup \
  /usr/lib/jvm \
  /usr/local/julia* \
  /usr/local/n \
  /usr/local/share/chromium \
  /usr/local/share/vcpkg \
  >/dev/null 2>&1 || true

# Remove unnecessary packages
sudo apt-get remove -y \
  azure-cli \
  google-cloud-sdk \
  firefox \
  google-chrome-stable \
  microsoft-edge-stable \
  mysql* \
  mongodb-org* \
  dotnet* \
  php* \
  >/dev/null 2>&1 || true

# Clean up package system
sudo apt-get autoremove -y >/dev/null 2>&1
sudo apt-get clean -y >/dev/null 2>&1

# Clean up package caches and data
sudo rm -rf \
  /var/lib/docker/* \
  /var/lib/gems/* \
  /var/lib/apt/lists/* \
  /var/cache/* \
  /var/lib/snapd \
  >/dev/null 2>&1 || true

# Print final disk space usage and difference
df -h / | awk -v before="$(df -h / | awk 'NR==2 {print $3}')" \
          'NR==2 {printf "After cleanup: %s used, %s free (freed %s)\n", 
                  $3, $4, substr(before,1,length(before)-1) - substr($3,1,length($3)-1) "G"}'