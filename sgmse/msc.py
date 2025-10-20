import torch
import torch.nn as nn
from einops import rearrange
from math import sqrt


class MSC(nn.Module):
    def __init__(self,dim,num_heads=8,topk=True,kernel = [3,5,7],s = [1,1,1],pad = [1,2,3],
                 qkv_bias=False,qk_scale=None,attn_drop_ratio=0.,proj_drop_ratio=0.,k1 = 2, k2 =3):
        super(MSC, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q    = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv   = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.k1   = k1
        self.k2   = k2
        
        self.attn1 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        # self.attn3 = torch.nn.Parameter(torch.tensor([0.3]), requires_grad=True)

        self.avgpool1 = nn.AvgPool2d(kernel_size=kernel[0],stride=s[0],padding=pad[0])
        self.avgpool2 = nn.AvgPool2d(kernel_size=kernel[1],stride=s[1],padding=pad[1])
        self.avgpool3 = nn.AvgPool2d(kernel_size=kernel[2],stride=s[2],padding=pad[2])

        self.layer_norm = nn.LayerNorm(dim)
        
        self.topk = topk # False True

    def forward(self, x,y):
        # x0 = x
        y1 = self.avgpool1(y)
        y2 = self.avgpool2(y)
        y3 = self.avgpool3(y)
        # y = torch.cat([y1.flatten(-2,-1),y2.flatten(-2,-1),y3.flatten(-2,-1)],dim = -1)
        y = y1+y2+y3
        y = y.flatten(-2,-1)
    
        y = y.transpose(1, 2)
        
        # 动态适应LayerNorm的维度（仅在第一次维度不匹配时创建）
        actual_dim = y.shape[-1]
        if actual_dim != self.layer_norm.normalized_shape[0]:
            # 如果实际维度与LayerNorm期望的维度不匹配，动态创建新的LayerNorm层
            # 使用缓存机制，避免重复创建
            if not hasattr(self, '_cached_ln_dim') or self._cached_ln_dim != actual_dim:
                self.layer_norm = nn.LayerNorm(actual_dim).to(y.device)
                self._cached_ln_dim = actual_dim  # 缓存当前LayerNorm维度
        
        y = self.layer_norm(y)
        x = rearrange(x,'b c h w -> b (h w) c')
        # y = rearrange(y,'b c h w -> b (h w) c')
        B, N1, C_y = y.shape
        B, N, C_x = x.shape
        
        # 动态适应线性层的维度（仅在第一次维度不匹配时创建）
        if C_y != self.kv.in_features:
            # 如果实际输入维度与kv线性层期望的维度不匹配，动态创建新的线性层
            # 使用缓存机制，避免重复创建
            if not hasattr(self, '_cached_dim') or self._cached_dim != C_y:
                self.kv = nn.Linear(C_y, C_y * 2, bias=self.kv.bias is not None).to(y.device)
                self.q = nn.Linear(C_x, C_x, bias=self.q.bias is not None).to(y.device)
                # proj层的输入维度需要根据head_dim是否匹配来决定
                proj_input_dim = self.num_heads * (C_y // self.num_heads) if C_x // self.num_heads != C_y // self.num_heads else C_x
                self.proj = nn.Linear(proj_input_dim, C_x, bias=self.proj.bias is not None).to(y.device)
                self._cached_dim = C_y  # 缓存当前维度
        
        # print(y.shape)
        kv  = self.kv(y).reshape(B, N1, 2, self.num_heads, C_y // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q   = self.q(x).reshape(B, N, self.num_heads, C_x // self.num_heads).permute(0,2,1,3)
        
        # 注意力计算：q [B, num_heads, N, C_x // num_heads] @ k [B, num_heads, C_y // num_heads, N1]
        # 需要确保 C_x // num_heads == C_y // num_heads
        # 如果维度不匹配，我们需要调整q的维度以匹配k
        head_dim_x = C_x // self.num_heads
        head_dim_y = C_y // self.num_heads
        
        if head_dim_x != head_dim_y:
            # 如果头维度不匹配，我们需要投影q到与k相同的维度
            # 这里我们使用一个线性层将q从head_dim_x投影到head_dim_y
            if not hasattr(self, '_cached_proj_q') or self._cached_proj_q.in_features != head_dim_x or self._cached_proj_q.out_features != head_dim_y:
                self.proj_q = nn.Linear(head_dim_x, head_dim_y, bias=False).to(x.device)
                self._cached_proj_q = self.proj_q
            q = self.proj_q(q)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
    
        # print(self.k1,self.k2)    
        mask1 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=int(N1/self.k1), dim=-1, largest=True)[1]
        # print(index[0,:,48])
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = (attn1 @ v)

        mask2 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=int(N1/self.k2), dim=-1, largest=True)[1]
        # print(index[0,:,48])
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        out2 = (attn2 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 #+ out3 * self.attn3
        # out = out1 * self.attn1 + out2 * self.attn2

        x = out.transpose(1, 2).reshape(B, N, self.num_heads * head_dim_y if head_dim_x != head_dim_y else C_x)
   
        x = self.proj(x)
        x = self.proj_drop(x)
        hw = int(sqrt(N))
        x = rearrange(x,'b (h w) c -> b c h w',h=hw,w=hw)
        # x = x + x0
        return x
    
if __name__ == "__main__":
    # 模拟输入参数
    batch_size = 1
    channels = 32  # 必须能被 num_heads 整除，默认 num_heads=8
    height = 64
    width = 64

    # 创建输入张量
    x = torch.randn(batch_size, channels, height, width)
    y = torch.randn(batch_size, channels, height, width)

    # 实例化模型
    model = MSC(dim=channels, num_heads=8, topk=True)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    y = y.to(device)
    model = model.to(device)

    # 前向传播
    output = model(x, y)

    # 打印模型结构
    print(model)
    print("微信公众号:AI缝合术")

    # 打印输入输出形状
    print("输入 x 形状:", x.shape)   # [B, C, H, W]
    print("输入 y 形状:", y.shape)   # [B, C, H, W]
    print("输出形状   :", output.shape) # [B, C, H, W]