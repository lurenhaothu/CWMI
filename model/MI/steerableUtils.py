import math 
import torch
from torch.autograd import Variable
from model.MI.config import *
#from config import *
import time
from concurrent.futures import ThreadPoolExecutor

def L( r ):
	if   r <= math.pi / 4:
		return 2 
	elif r >= math.pi / 2:
		return 0 
	else:
		return 2 * math.cos( math.pi / 2 * math.log( 4 * r / math.pi ) / math.log( 2 ) ) 

def H( r ):
	if   r <= math.pi / 4:
		return 0 
	elif r >= math.pi / 2:
		return 1 
	else:
		return     math.cos( math.pi / 2 * math.log( 2 * r / math.pi ) / math.log( 2 ) ) 

def G( t, k, K ):

	t0 = math.pi * k / K 
	aK = 2**(K-1) * math.factorial(K-1) / math.sqrt( K * math.factorial( 2 * (K-1) ) )

	if (t - t0) > (math.pi/2):
		return G( t - math.pi, k, K ) 
	elif (t - t0 ) < (-math.pi/2):
		return G( t + math.pi, k, K )
	else:
		return aK * (math.cos( t - t0 ))**(K-1)

def S( t, k, K ):

	t0 = math.pi * k / K 
	dt = abs(t-t0)

	if   dt <  math.pi/2:
		return 1 
	elif dt == math.pi/2:
		return 0
	else:
		return -1 

def L0( r ):
	return L( r/2 ) / 2 

def H0( r ):
	return H( r/2 )

def polar_map( s ):

	x = torch.linspace(        0, math.pi, s[1] ).view( 1, s[1] ).expand( s )
	if s[0] % 2 == 0 :
		y = torch.linspace( -math.pi, math.pi, s[0]+1 ).narrow(0,1,s[0])
	else:
		y = torch.linspace( -math.pi, math.pi, s[0]   )
	y = y.view( s[0], 1 ).expand( s ).mul( -1 )

	r = ( x**2 + y**2 ).sqrt()
	t = torch.atan2( y, x )

	return r, t 

def S_matrix( K, s ):

	_, t = polar_map( s )
	sm = torch.Tensor( K, s[0], s[1] )

	for k in range( K ):
		t0 = math.pi * k / K 
		dt = torch.abs(t-t0)
		sm[k, :, :] = (dt < math.pi/2).to(torch.int) - (dt > math.pi/2).to(torch.int)
		#for i in range( s[0] ):
		#	for j in range( s[1] ):
		#		sm[k][i][j] = S( t[i][j], k, K )

	return sm 

def G_matrix( K, s ):

	_, t = polar_map( s ) 
	g = torch.Tensor( K, s[0], s[1] ) 

	for k in range( K ):
		t0 = math.pi * k / K 
		aK = 2**(K-1) * math.factorial(K-1) / math.sqrt( K * math.factorial( 2 * (K-1) ) )
		g[k, :, :] = aK * (torch.abs(torch.cos( t - t0 )))**(K-1)
		#for i in range( s[0] ):
		#	for j in range( s[1] ):
		#		g[k][i][j] = G( t[i][j], k, K ) 

	return g 

def B_matrix( K, s ):

	g = G_matrix( K, s ) 

	r, _ = polar_map( s )
	h = r.apply_( H ).unsqueeze(0)

	return h * g 

def L_matrix( s ):

	r, _ = polar_map( s )

	return r.apply_( L )

def LB_matrix( K, s ):

	l = L_matrix( s ).unsqueeze(0) 
	b = B_matrix( K, s ) 

	return torch.cat( (l,b), 0 )

def HL0_matrix( s ):

	r, _ = polar_map( s )
	h = r.clone().apply_( H0 ).view( 1, s[0], s[1] ) 
	l = r.clone().apply_( L0 ).view( 1, s[0], s[1] )

	return torch.cat( ( h, l ), 0 )

def central_crop( x ):

	ns = [ x.size(-2)//2 , x.size(-1)//2 + 1 ]

	return x.narrow( -2, ns[1]-1, ns[0] ).narrow( -1, 0, ns[1] )

def cropped_size( s ):

	return [ s[0]//2 , s[1]//2 + 1 ]

def L_matrix_cropped( s ):

	l = L_matrix( s ) 

	ns = cropped_size( s ) 

	return l.narrow( 0, ns[1]-1, ns[0] ).narrow( 1, 0, ns[1] ) 

def freq_shift( imgSize, fwd ):
    ind = torch.LongTensor( imgSize ).to(device)
    sgn = 1 
    if fwd:
        sgn = -1 
    for i in range( imgSize ):
        ind[i] = (i + sgn*((imgSize-1)//2) ) % imgSize

    return Variable( ind ) 
