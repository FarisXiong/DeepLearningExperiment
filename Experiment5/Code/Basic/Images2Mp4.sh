ffmpeg  -r 5 -pattern_type glob -i '../../Trains/Images/GAN/*.png' -c:v libx264 ../../Trains/Videos/GAN.mp4
ffmpeg  -r 5 -pattern_type glob -i '../../Trains/Images/WGAN/*.png' -c:v libx264 ../../Trains/Videos/WGAN.mp4
ffmpeg  -r 5 -pattern_type glob -i '../../Trains/Images/WGAN-GP/*.png' -c:v libx264 ../../Trains/Videos/WGAN-GP.mp4