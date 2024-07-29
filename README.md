# README.md

## Build

Build with cmake. Cmake's FetchContent is used to fetch all dependencies.

```sh
$ cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
$ cmake --build build --target metal-raytracer -- -j
```

## Run

Run the raytracer by providing it a glTF file.

```sh
$ ./build/metal-raytracer assets/Sponza.glb
```
