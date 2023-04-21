ModuleDict(
  (wte): Embedding(50257, 1280)
  (wpe): Embedding(1024, 1280)
  (drop): Dropout(p=0.1, inplace=False)
  (h): ModuleList(
    (0-35): 36 x Block(
      (ln_1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
      (attn): CausalSelfAttention(
        (c_attn): Linear(in_features=1280, out_features=3840, bias=True)
        (c_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (attn_dropout): Dropout(p=0.1, inplace=False)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
      (ln_2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
      (mlp): ModuleDict(
        (c_fc): Linear(in_features=1280, out_features=5120, bias=True)
        (c_proj): Linear(in_features=5120, out_features=1280, bias=True)
        (act): NewGELU()
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (ln_f): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
)
