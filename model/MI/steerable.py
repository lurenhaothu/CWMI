import torch
import torch.nn as nn

from model.MI.steerableUtils import *  
from model.MI.config import *

#from steerableUtils import *  
#from config import *

import time

class SteerablePyramid(nn.Module):

    def __init__(self, imgSize, K=4, N=4, hilb=False, includeHF=True ):
        super(SteerablePyramid, self).__init__()

        size = [ imgSize, imgSize//2 + 1 ]
        self.hl0 = HL0_matrix( size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

        self.l = []
        self.b = []
        self.s = []

        self.K    = K 
        self.N    = N 
        self.hilb = hilb
        self.includeHF = includeHF 

        self.indF = [ freq_shift( size[0], True  ) ] 
        self.indB = [ freq_shift( size[0], False ) ] 


        for n in range( self.N ):

            l = L_matrix_cropped( size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
            b = B_matrix(      K, size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
            s = S_matrix(      K, size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

            self.l.append( l.div_(4) )
            self.b.append( b )
            self.s.append( s )

            size = [ l.size(-2), l.size(-1) ]

            self.indF.append( freq_shift( size[0], True  ) )
            self.indB.append( freq_shift( size[0], False ) )


    def forward(self, x):

        # fftfull = torch.rfft(x,2)
        fftfull = torch.fft.rfft2(x)
        #xreal = fftfull[... , 0]
        #xim = fftfull[... ,1]
        xreal = fftfull.real
        xim = fftfull.imag
        x = torch.cat((xreal.unsqueeze(1), xim.unsqueeze(1)), 1 ).unsqueeze( -3 )
        x = torch.index_select( x, -2, self.indF[0] )

        # print(self.hl0.shape, self.hl0)
        x   = self.hl0 * x 
        h0f = x.select( -3, 0 ).unsqueeze( -3 )
        l0f = x.select( -3, 1 ).unsqueeze( -3 )
        lf  = l0f 

        output = []

        for n in range( self.N ):

            bf = self.b[n] *               lf 
            lf = self.l[n] * central_crop( lf ) 
            if self.hilb:
                hbf = self.s[n] * torch.cat( (bf.narrow(1,1,1), -bf.narrow(1,0,1)), 1 )
                bf  = torch.cat( ( bf , hbf ), -3 )
            if self.includeHF and n == 0:
                bf  = torch.cat( ( h0f,  bf ), -3 )

            output.append( bf )

        output.append( lf  ) 

        for n in range( len( output ) ):
            output[n] = torch.index_select( output[n], -2, self.indB[n] )
            sig_size = output[n].shape[-2]
            output[n] = torch.stack((output[n].select(1,0), output[n].select(1,1)),-1)
            # output[n] = torch.fft.irfft( output[n], 2, signal_sizes = [sig_size, sig_size])
            # print(output[n].shape)
            output[n] = torch.fft.irfft2( torch.complex(output[n][...,0], output[n][...,1]), s=[sig_size, sig_size], dim=(-2, -1))
            # print(output[n].shape)

        if self.includeHF:
            output.insert( 0, output[0].narrow( -3, 0, 1                    ) )
            output[1]       = output[1].narrow( -3, 1, output[1].size(-3)-1 )

        for n in range( len( output ) ):
            if self.hilb:
                if ((not self.includeHF) or 0 < n) and n < len(output)-1:
                    nfeat = output[n].size(-3)//2
                    o1 = output[n].narrow( -3,     0, nfeat ).unsqueeze(1)
                    o2 = output[n].narrow( -3, nfeat, nfeat ).unsqueeze(1)
                    output[n] = torch.cat( (o1, o2), 1 ) 
                else:
                    output[n] = output[n].unsqueeze(1)
        
        for i in range(len(output)):
            output[i] = output[i].squeeze(1)

        return output


def testSteerable():

    imgSize = 128 
    network = SteerablePyramid( imgSize=imgSize, K=4, N=4 )
    x = torch.randn((1,1,imgSize,imgSize),requires_grad=True, device=device)    

    y = network( x ) 

    for i in range(len(y)): 
        y[i].backward( y[i], retain_graph=True)
    
    z = x.grad


    print( '\ninput size' )
    print( x.size() ) 

    print( '\noutput size' )
    for i in range(len(y)):
        img = y[i].data#.select(0,0)#.select(0,0)           
        # img = img.view( -1, img.size(-2), img.size(-1) )
        print( i, img.size() )
        # for j in range(img.size(0)):
        #     scale = 127.5 / max( img[j].max(), - img[j].min() )
        #     im = Image.fromarray( img[j].mul(scale).add(127.5).cpu().numpy().astype(numpy.uint8) )
        #     im.save( 'steerable_' + str(i) + '_' + str(j) + '.png' )

    print( '\nreconstruction size')
    print( z.size() ) 


    print( '\nreconstruction error', x.dist( z ).data.squeeze() ) 

if __name__ == "__main__":
    testSteerable()
