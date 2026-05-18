#!/bin/bash
ffmpeg -framerate 2 -pattern_type glob -i "gauss_evolved_n*.png" output.mp4