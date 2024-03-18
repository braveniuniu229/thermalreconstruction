import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class IncontextViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dim_head=64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, padding=1,dtype=torch.float32)
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )
        #随机初始化septoken，维度是图片嵌入的维度
        self.sep_token = nn.Parameter(torch.randn(dim))
        self.ans_token = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img , examples):

        device = img.device
        exp_num = examples.shape[1]
        batch_size = img.shape[0]
        img = self.conv1(img)
        processed_examples = torch.zeros(batch_size,exp_num,2,3,64,64,device=device)
        for i in range(exp_num):
            processed_examples[:, i, 0] = self.conv1(examples[:, i, 0])
            processed_examples[:, i, 1] = self.conv1(examples[:, i, 1])

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)
        sep_token = self.sep_token.unsqueeze(0).unsqueeze(1).repeat(batch_size,1 , 1).to(device)
        ans_token = self.ans_token.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1).to(device)
        prompt = [sep_token]
        for i in range(exp_num):  # 假设examples的形状为 [batch_size, n_examples, 2, C, H, W]
            example_input = self.to_patch_embedding(processed_examples[:, i, 0]) + self.pos_embedding.to(device, dtype=x.dtype)
            example_output = self.to_patch_embedding(processed_examples[:, i, 1]) + self.pos_embedding.to(device, dtype=x.dtype)
            prompt.append(example_input)
            prompt.append(ans_token)  # 在输入和输出之间添加<SEP>Token
            prompt.append(example_output)
            prompt.append(sep_token)  # 在不同的示例之间添加<SEP>Token
        prompt_tensor =torch.cat(prompt,dim=1)
        x = torch.cat((x,prompt_tensor),dim=1)
        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)
def countPara(m:nn.Module):
    return sum(p.numel() for p in m.parameters() if m.requires_grad_())
if __name__=="__main__":
    device = torch.device("cuda")
    img = torch.randn(5,2,64,64).to(device)
    samples = torch.randn(5,2,2,2,64,64).to(device)
    model = IncontextViT(image_size=(64, 64), patch_size=(4, 4), num_classes=4096, dim=256, depth=7, heads=8, mlp_dim=4).to(device).to(torch.float32)
    out = model(img,samples)
    print(countPara(model),out.shape)

