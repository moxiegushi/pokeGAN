# pokeGAN
Creats new kinds of pokemons using WGAN! (DCGAN is also supported)
## dependence
```
cv2
tensorflow( >=1.0)
scipy
numpy
```

Possible installation of dependencies:
```
conda create -n pokemon python=3.7    # creating a "pokemon" environment.
conda activate pokemon                # switching to this "pokemon" environment.
conda install numpy scipy tensorflow                             # installing dependencies. 
conda install --channel https://conda.anaconda.org/menpo opencv3 # installation of opencv
pip install jupyter --user      # installation of Jupyter; be careful, the installation path might be specified in comments.
```
## how to use
(you might need 
```
conda activate pokemon
```
if you have created an environment as above)
```
git clone ‘https://github.com/moxiegushi/pokeGAN.git’
cd pokeGAN
python resize.py
python RGBA2RGB.py
python pokeGAN.py
```
## example pokemons
![image1](https://github.com/moxiegushi/pokeGAN/raw/master/images/Notes_1500532347861.jpeg)

![image2](https://github.com/moxiegushi/pokeGAN/raw/master/images/Notes_1500532371830.jpeg)

It is difficult to train a GAN perfectly, and as you can see someimages are meaningless.
I'm very new to this. Please let me know if there are bugs :)
