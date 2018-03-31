# simple-gan-demo
Simple GAN implementation that outputs a video showing changes in discriminator and generator output across training steps.

Just run `python demo.py` to run the script, which trains a generator network to map a latent Gaussian distribution (mean 0, variance 1) to another prior Gaussian distribution (mean 3, variance 1).

The generator and discriminator networks are trained for 300 steps by default.

Both the generator and discriminator comprise 2 hidden layers with 128 nodes each. The script uses the Adam optimizer with a learning rate of 0.001.

The output video should show the generator output (blue) moving towards the original prior (green). The discriminator output (red) should also go lower in the generator distribution and higher in the original prior distribution.

![alt text](https://raw.githubusercontent.com/greentfrapp/simple-gan-demo/master/images/start.png "Start of training")

*Video cap at the start of the training. Generator output is far from original prior.*

Eventually, the generator output distribution should match the original prior distribution.

![alt text](https://raw.githubusercontent.com/greentfrapp/simple-gan-demo/master/images/end.png "End of training")

*Video cap at the end of the training. Generator output matches original prior.*