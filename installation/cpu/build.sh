docker build --build-arg UID="$(id -u)" \
             --build-arg USER="$(id -un)" \
             -t "$(id -un)/behavior-transformer" -f installation/cpu/Dockerfile .
