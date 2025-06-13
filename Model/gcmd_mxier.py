import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange


class FeedForward(nn.Module):
    '''
    B:batch
    K:links/nodes
    L:length
    H:hidden
    in(B K L H)
    out(B k L H)
    '''
    def __init__(
            self,
            dim,
            hidden_dim,
            dropout=0.
            ):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        x=self.net(x)
        return x

class MixerBlock(nn.Module):
    '''
    B:batch
    K:links/nodes
    L:length
    H:hidden
    in(B K L H)
    out(B k L H)
    '''
    def __init__(
                self,
                sequence_len,
                sequence_hid,
                dim,dropout=0.
                ):
        super().__init__()
        
        self.t_mixer=nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('B K L H -> B K H L'),
            FeedForward(sequence_len,sequence_hid,dropout),
            Rearrange('B K H L -> B K L H')
         )
        
        # self.f_mixer=nn.Sequential(
        #     nn.LayerNorm(dim),
        #     FeedForward(dim,dim,dropout)
        # )

    def forward(self,x):
        x = x+self.t_mixer(x)
        # x = x+self.f_mixer(x)
        return x


class GCNLayer(nn.Module):
    '''
    B:batch
    K:links/nodes
    L:length
    in(B K L)
    out(B k L)
    '''    
    def __init__(
            self,
            in_features,
            out_features
            ):
        super(GCNLayer, self).__init__()
        self.conv =nn.Sequential(
            nn.Linear(in_features,out_features),
            nn.GELU(),
            Rearrange('B K L -> B L K'),
            ) 
    def forward(self, x, A):
        x = self.conv(x)
        x = x@A
        x = rearrange(x, 'B L K -> B K L')
        return x

class MVMD(nn.Module):
    '''
    K:links/nodes
    L:length
    in(K L)
    out(k L)
    '''    
    def __init__(
            self,
            alpha, 
            tau,
            K,
            DC,
            init,
            tol,
            max_N
            ):
            super(MVMD, self).__init__()
            self.alpha=alpha
            self.tau=tau
            self.K=K
            self.DC=DC
            self.init=init
            self.tol=float(tol)
            self.max_N=max_N

    def forward(self,signal):
        alpha=self.alpha
        tau=self.tau
        K=self.K
        DC=self.DC
        init=self.init
        tol=self.tol
        max_N=self.max_N
        C, T = signal.shape
        fs = 1 / float(T)
        f_mirror = torch.zeros(C, 2*T)
        f_mirror[:,0:T//2] = torch.flip(signal[:,0:T//2], dims=[-1]) 
        f_mirror[:,T//2:3*T//2] = signal
        f_mirror[:,3*T//2:2*T] = torch.flip(signal[:,T//2:], dims=[-1])
        f = f_mirror

        T = float(f.shape[1])
        t = torch.linspace(1/float(T), 1, int(T))
        freqs = t - 0.5 - 1/T
        N = max_N
        Alpha = alpha * torch.ones(K, dtype=torch.cfloat)
        f_hat = torch.fft.fftshift(torch.fft.fft(f))
        f_hat_plus = f_hat
        f_hat_plus[:, 0:int(int(T)/2)] = 0
        u_hat_plus = torch.zeros((N, len(freqs), K, C), dtype=torch.cfloat)
        omega_plus = torch.zeros((N, K), dtype=torch.cfloat)
                            
        if (init == 1):
            for i in range(1, K+1):
                omega_plus[0,i-1] = (0.5/K)*(i-1)
        elif (init==2):
            omega_plus[0,:] = torch.sort(torch.exp(torch.log(fs)) +
            (torch.log(0.5) - torch.log(fs)) * torch.random.rand(1, K))
        else:
            omega_plus[0,:] = 0

        if (DC):
            omega_plus[0,0] = 0

        lamda_hat = torch.zeros((N, len(freqs), C), dtype=torch.cfloat)

        uDiff = tol+1e-16
        n = 1 
        sum_uk = torch.zeros((len(freqs), C))

        T = int(T)

        while uDiff > tol and n < N:
            k = 1
            sum_uk = u_hat_plus[n-1,:,K-1,:] + sum_uk - u_hat_plus[n-1,:,0,:]
            for c in range(C):
                u_hat_plus[n,:,k-1,c] = (f_hat_plus[c,:] - sum_uk[:,c] - 
                lamda_hat[n-1,:,c]/2) \
            / (1 + Alpha[k-1] * torch.square(freqs - omega_plus[n-1,k-1]))
    
            if DC == False:
                omega_plus[n,k-1] = torch.sum(torch.mm(freqs[T//2:T].unsqueeze(0), 
                                torch.square(torch.abs(u_hat_plus[n,T//2:T,k-1,:])))) \
                / torch.sum(torch.square(torch.abs(u_hat_plus[n,T//2:T,k-1,:])))

            for k in range(2, K+1):
                sum_uk = u_hat_plus[n,:,k-2,:] + sum_uk - u_hat_plus[n-1,:,k-1,:]
                for c in range(C):
                    u_hat_plus[n,:,k-1,c] = (f_hat_plus[c,:] - sum_uk[:,c] - 
                lamda_hat[n-1,:,c]/2) \
                / (1 + Alpha[k-1] * torch.square(freqs-omega_plus[n-1,k-1]))
                omega_plus[n,k-1] = torch.sum(torch.mm(freqs[T//2:T].unsqueeze(0),
                    torch.square(torch.abs(u_hat_plus[n,T//2:T,k-1,:])))) \
                    /  torch.sum(torch.square(torch.abs(u_hat_plus[n,T//2:T:,k-1,:])))
            lamda_hat[n,:,:] = lamda_hat[n-1,:,:]
            n = n + 1
            uDiff = 2.2204e-16

            for i in range(1, K+1):
                uDiff=uDiff+1 / float(T) * torch.mm(u_hat_plus[n-1,:,i-1,:] - u_hat_plus[n-2,:,i-1,:], 
                                                    ((u_hat_plus[n-1,:,i-1,:]-u_hat_plus[n-2,:,i-1,:]).conj()).conj().T)
                
            uDiff = torch.sum(torch.abs(uDiff))

        N = min(N, n)
        omega = omega_plus[0:N,:]
        u_hat = torch.zeros((T,K,C), dtype=torch.cfloat)
        for c in range(C):
            u_hat[T//2:T,:,c] = torch.squeeze(u_hat_plus[N-1,T//2:T,:,c])
            second_index = list(range(1,T//2+1))
            second_index.reverse()
            u_hat[second_index,:,c] = torch.squeeze(torch.conj(u_hat_plus[N-1,T//2:T,:,c]))
            u_hat[0,:,c] = torch.conj(u_hat[-1,:,c])
        u = torch.zeros((K,len(t),C), dtype=torch.cfloat)

        for k in range(1, K+1):
            for c in range(C):
                u[k-1,:,c]  = (torch.fft.ifft(torch.fft.ifftshift(u_hat[:,k-1,c]))).real
        u = u[:,T//4:3*T//4,:]
        u_hat = torch.zeros((T//2,K,C), dtype=torch.cfloat)

        for k in range(1, K+1):
            for c in range(C):
                u_hat[:,k-1,c] = torch.fft.fftshift(torch.fft.fft(u[k-1,:,c])).conj()
        u = torch.fft.ifftshift(u, dim=-1)
        
        return (u.real, u_hat, omega)



class GCMDMixer(nn.Module):

    def __init__(
                self,
                 sequence_len,
                 l_sequence_hid,
                 g_sequence_hid,
                 input_dim,
                 emb_dim,
                 split,
                 adj,
                 modes,
                 dropout=0.
                 ):
        super(GCMDMixer, self).__init__()
        self.split=split
        self.sequence_len=sequence_len
        self.x_embdding= nn.Linear(input_dim,emb_dim)

        self.local_mixer =MixerBlock(int(sequence_len//split),l_sequence_hid,emb_dim,dropout)
        self.global_mixer = MixerBlock(sequence_len,g_sequence_hid,emb_dim,dropout)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.pooling=nn.Linear(emb_dim,1)
        self.weighted_avg=nn.Linear(modes,1)
        self.A=adj
        self.conv1=GCNLayer(sequence_len,sequence_len)
        self.conv2=GCNLayer(sequence_len,sequence_len)


    def forward(self,x,md):
        '''
        B:batch
        K:links/nodes
        L:length
        F:feature
        H:hidden
        M:input modes
        x shape(B,K,L,F)
        md shape(B,K,L,M)
        A shape(K,K)
        '''
        assert(self.sequence_len%self.split==0)
        x=self.x_embdding(x)
        x_l=x
        l_s=self.sequence_len//self.split
        s_p=self.split
        x_l = rearrange(x_l, 'B K (S P) H -> (B P) K S H ',S=l_s,P=s_p)
        x_l = self.local_mixer(x_l)
        x_l = rearrange(x_l, '(B P) K S H -> B K (S P) H',S=l_s,P=s_p)
        x_g = self.global_mixer(x)
        x = x_g + self.alpha*(x_l - x_g)
        x=self.pooling(x).squeeze(dim=3)
        
        p=self.weighted_avg(md).squeeze(dim=3)
        p=self.conv1(p,self.A)
        p=self.conv2(p,self.A)

        x=x+p

        return x