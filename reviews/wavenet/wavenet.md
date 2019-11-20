# WAVENET: A GENERATIVE MODEL FOR RAW AUDIO
date 19.09.2016
paper on [arxiv][arx]
____________________________
WaveNet  - autoregressive CNN based model for generating raw audio files. It is able to generate audio conditioned on speaker and text (TTS).
## Model overview
### Causal Dilated Convolutions
All convolutions used it model was causal. By causal, authors mean that model cannot violate the ordering in which we model the data : the prediction $p(x_t |x_{t-1}, \dots, x_1)$ at timestamp t doesn't depend on future timestamps $x_{t+1}, \dots, x_T$.  It could be implemented by shifting the output of a normal convolution by a few timesteps (this shifting is equal to zero padding left side of input  to the convolution with kernel size $k$ and dilation rate $d$ by $(k-1) * d$ zeros). Note that convolutions with kernel size $=1$ are not shifted and with this padding convolution preserves input length.
 ![enter image description here](https://lh3.googleusercontent.com/9ZhT4xHtaytOL4OHnyC26Ba2-BV81CTjpJKeoXBrekZevLBccQTIloDk0lCpn6uQkpCmdjCdQVMt)
 On image above convolution with kernel size $=2$ and dilation rate $=1$ is done. 
Since raw audio files represented by thousands of elements per second, model must have a huge receptive field to capture dependencies. Dilated convolutions was claimed to do this. Authors proposed stacking convolutions with doubling dilation rate till some limit and then repeat this stacking several times. For example in paper following sequence of dilation rate was used:
$$
1, 2, 4, \dots, 512, 1, 2, 4, \dots, 512, 1, 2, 4, \dots, 512.
$$ 
If all convolutions was with the same kernel_size then receptive field for output would be equal to (kernel_size $-1) *$ sum(dilation rates) $+ 1$. For example, if kernel_size $=3$ then receptive field equals to 6139.
Example of convolutions with kernel size $=2$ and dilation rates growing from 1 to 8 bottom-up could be seen on image below.
![enter image description here](https://lh3.googleusercontent.com/DuDlKKcOt0WFDqUSXm3pWdvWo7BmKGDBIbxCqvFQzTaDtLDiUAqphs-cbklPa8MP-4rAxBfxQ6na)
_____________

## Model architecture
Model consists of causal dilated convolutions with skip and residual connections. 
![enter image description here](https://lh3.googleusercontent.com/VVP44jnF3-lfsFbfaFze4VV-xee7zD69Ir-LSdSbnuGd0aMv4nctl-pndp2eGl71blKewR84ZTcI)
Model could be conditioned on speaker or text via extending gated activations.  Vanilla gated activation ( $*$ is convolution, $\odot$ is Hadamard  product):
$$
z = tanh(W_f *x) \odot sigmoid(W_g *x).
$$
Let $h$ be user embedding, e.g. one hot vector (any else?) then 
$$
z = tanh(W_f *x + V_fh) \odot sigmoid(W_g *x + V_gh).
$$
In case of text conditioning, we have some linguistic features $h_t$, with length much smaller than raw audio signal. We need to upsample with features to match audio length. It could be done via stacking linguistic features $h_t$ or via applying some convolution transposed network $f$ to it(not properly covered, have questions about it). Then $y$ features with the same length as audio signal and:
$$
z = tanh(W_f *x + V_f * y) \odot sigmoid(W_g *x + V_g * y).
$$

_________
## Data preprocessing.
Since raw audio signals at each timestamp is represented via 16-bit integer, model must predict one of 65536. Calculating softmax with such a big number of classes is numerically unstable operation. To avoid this problem authors proposed quantizing classes to 256 values using  mu-law transformation.
## Training
During training since out convolutions are casual we can process data one time, using as input one hot encoded vectors of audio.
## Testing 
During testing we predict one label as a time, than one hot encode it and use as input for forward prediction. Its not covered in article but as far as as i understand using only last  N elements (N equals to receptive field of model) makes sense for predicting next element.
## Experiments
Not done yet due, i want to find answers to following questions before.
# Questions
1. In section 3.2 authors uses words "linguistic features". Its unclear to me what is it and how it was obtained.
2. Authors also used external models for prediciting logarithmic fundamental frequency and phone durations. Not clear what models?
3. Just to make sure,  Is fundamental frequency is property of speaker? In article "External models predicting $log F_0$ values and phone durations from linguistic features were also trained for each language" its look like language property.
