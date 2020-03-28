import theano
import lasagne
import theano.tensor as T
from theano import shared
from lasagne import init
import numpy as np
from lasagne.layers import Layer,MergeLayer,InputLayer
import theano.tensor.signal.conv as TS
import lasagne.theano_extensions.conv as conv
import math

def flip(x, dim):
    xsize = x.shape
    # print xsize
    # dim = x.dim() + dim if dim < 0 else dim
    # print dim
    # x = x.contiguous()
    x = x[None,:]
    # x = T.reshape(x,(-1, xsize))
    x = T.reshape(x,(x.shape[0], x.shape[1], -1))[:,T.arange(x.shape[1]-1, 
                      -1, -1), :]
    return T.reshape(x,xsize)

def sinc(band,t_right):
    y_right= T.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    # print 'band = ' + str(band)
    # print 't_right = ' + str(t_right)
    y_left= flip(y_right,0)   
    # y_left = y_right
    y=T.concatenate([y_left,T.ones(1),y_right])
    return y


class SincConv(Layer):

    def __init__(self, incoming, fs , N_filt, Filt_dim,**kwargs):
        super(SincConv, self).__init__(incoming, **kwargs)
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10**(mel_points / 2595) - 1)) # Convert Mel to Hz
        b1=np.roll(f_cos,1)
        b2=np.roll(f_cos,-1)
        b1[0]=30
        b2[-1]=(fs/2)-100


        self.N_filt=N_filt
        self.Filt_dim=Filt_dim
        self.fs = fs
        # self.W=self.add_param(W,(self.N_filt,self.Filt_dim))
        self.freq_scale=fs*1.0
        self.filt_b1 = self.add_param(T.as_tensor_variable(b1/self.freq_scale),shape=b1.shape)
        self.filt_band = self.add_param(T.as_tensor_variable((b2-b1)/self.freq_scale),shape=(b2-b1).shape)
        # self.filters=self.add_param(T.as_tensor_variable(np.zeros((self.N_filt,self.Filt_dim))),(self.N_filt,self.Filt_dim))
        # self.t_right=self.add_param(T.as_tensor_variable(np.linspace(1, (self.Filt_dim-1)/2, num=int((self.Filt_dim-1)/2))/self.fs),(int((self.Filt_dim-1)/2),))
        # self.window = self.add_param(T.cast(T.as_tensor_variable(0.54-0.46*T.cos(2*math.pi*n/self.Filt_dim)),'float32'),(self.Filt_dim,))

    def get_output_shape_for(self, input_shape):
		return input_shape[0],self.N_filt,input_shape[2]-self.Filt_dim+1

    def get_output_for(self, x, **kwargs):
    	filters=T.zeros((self.N_filt,self.Filt_dim))
        # filters = self.W
    	N=self.Filt_dim
    	t_right=T.as_tensor_variable(np.linspace(1, (self.Filt_dim-1)/2, num=int((self.Filt_dim-1)/2))/self.fs)
    	min_freq=50.0
    	min_band=50.0
    	filt_beg_freq=T.abs_(self.filt_b1)+min_freq/self.freq_scale
        # filt_beg_freq *= 0.0
    	filt_end_freq=filt_beg_freq+(T.abs_(self.filt_band)+min_band/self.freq_scale)
        # filt_end_freq = filt_end_freq*0.0 + self.fs/2
    	n=T.as_tensor_variable(np.linspace(0, N, num=N))
    	# print n
    	window=T.cast(T.as_tensor_variable(0.54-0.46*T.cos(2*math.pi*n/self.Filt_dim)),'float32')
    	# window = T.cast(window,'float32')
    	# window = theano.shared(window,name='window')
    	# window=theano.shared(0.54-0.46*T.cos(2*math.pi*n/N),name='window',dtype='float32')
    	for i in range(self.N_filt):
    		low_pass1 = T.cast(2*filt_beg_freq[i],'float32')*sinc(T.cast(filt_beg_freq[i],'float32')*self.freq_scale,t_right)
    		low_pass2 = T.cast(2*filt_end_freq[i],'float32')*sinc(T.cast(filt_end_freq[i],'float32')*self.freq_scale,t_right)
    		band_pass=(low_pass2-low_pass1)
    		band_pass=band_pass/T.max(band_pass)
    		filters = T.set_subtensor(filters[i,:],band_pass*window)
            # filters = T.set_subtensor(filters[i,:],self.W[i,:])
    		# filters[i,:]=self.W[i,:]
        # filters = filters+0.0*self.W
        out=conv.conv1d_mc0(x, T.reshape(filters,(self.N_filt,1,self.Filt_dim)))
        # out=conv.conv1d_mc0(x, self.W)
        # print out.shape
        return T.abs_(out)

	

# input_var = T.tensor3('inputs')
# N_filt = 80
# Filt_dim=251
# filters=T.zeros((N_filt,1,Filt_dim))
# # out = T.signal.conv.conv2d(input_var, filters.reshape(N_filt,1,Filt_dim))
# # out = TS.conv2d(input_var,filters)
# # test_fn = theano.function([input_var],out)
# dummy_data = np.random.random([5,1,1000]).astype(np.float32)
# input_layer = InputLayer(dummy_data.shape, input_var)

# network = SincConv(input_layer,fs=16000,N_filt=N_filt,Filt_dim=Filt_dim)
# prediction = lasagne.layers.get_output(network)
# # prediction = network.get_output_for(network,input_var)
# test_fn = theano.function([input_var],prediction)

# out = test_fn(dummy_data)
# print out.shape


