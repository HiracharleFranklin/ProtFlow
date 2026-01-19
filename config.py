import ml_collections
from transformers import BertConfig

def create_config():
    config = ml_collections.ConfigDict()
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.
    optim.linear_warmup = 5_000
    optim.lr = 1e-4
    optim.warmup_lr = 0
    optim.weight_decay = 0.01
    optim.beta_1 = 0.9
    optim.beta_2 = 0.98
    optim.eps = 1e-6

    scheduler = config.scheduler = ml_collections.ConfigDict()
    scheduler.type = "cosine"


    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 1_000_000
    training.checkpoint_freq = 1_000
    training.generate_freq = 1_000
    training.eval_freq = 1_000
    training.batch_size = 64

    training.ode_sampling = False
    training.checkpoints_folder = '' #to be defined
    config.checkpoints_prefix = '' #to be defined

    loss = config.loss = ml_collections.ConfigDict()
    loss.ce_coef = 0.
    loss.x0_coef = 0.2

    refresh = config.refresh = ml_collections.ConfigDict()
    refresh.true = False
    refresh.use_pretrain = False
    refresh.prefix = '' #to be defined

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = 64
    validation.validation_iters = int(10_000 / validation.batch_size)
    validation.num_gen_texts = 1024

    fm = config.fm = ml_collections.ConfigDict()
    fm.reflow = False
    fm.reflow_generate_data = False
    fm.reflow_ckpt = '' #to be defined
    fm.reflow_datapath = '' #to be defined
    fm.reflow_datanum = 0 #to be defined
    fm.sample_std = 1
    fm.flow_mode = "1-rf"
    fm.ot_sampler_mode = "exact"
    fm.sigma = 0.1
    fm.reweight = False
    fm.m = 0.
    fm.s = 1.
    fm.solver = '' #to be defined, e.g. euler, rk4, dopri5
    fm.sensitivity = 'adjoint'
    fm.use_mask = False
    config.sampling_step = 25


    config.use_compress = True #True
    compress = config.compress = ml_collections.ConfigDict()
    compress.checkpoint = '' #to be defined
    compress.depth = 4
    compress.downproj_factor = 16
    compress.shorten_factor = 1
    compress.attn_resampling = True
    compress.updown_sample_type = "naive"
    compress.heads = 8
    compress.dim_head = 64
    compress.causal = False
    compress.norm_out = False
    compress.use_quantizer = "tanh"
    compress.n_e = 512
    compress.e_dim = 64
    compress.vq_beta = 0.25
    compress.enforce_single_codebook_per_position = False
    compress.fsq_levels = [8,8,8,8,8,8]
    compress.verbose = False

    model = config.model = ml_collections.ConfigDict()
    model.model_type = "bert"
    model.ema_rate = 0.9999
    model.embeddings_type = "encodings"
    model.dif_enc_type = "base"
    model.prediction = "x_0"
    model.loss = "L_x_0"
    model.hidden_size = 480
    model.compressed_hidden_size = 30
    model.hg_name = "facebook/esm2_t12_35M_UR50D"
    model.hg_name_hash = "esm2-35M"
    model.hidden_layers = 12
    model.intermediate_size = 3072
    model.attention_heads = 16

    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = 50
    data.dataset = '' #to be defined
    data.train_dataset_path = f'./data/{data.dataset}/train.fasta'
    data.test_dataset_path = f'./data/{data.dataset}/valid.fasta'

    
    data.enc_mean = f"./data/{data.dataset}/encodings-{model.hg_name_hash}-mean.pt"
    data.enc_std = f"./data/{data.dataset}/encodings-{model.hg_name_hash}-std.pt"
    data.enc_max = f"./data/{data.dataset}/encodings-{model.hg_name_hash}-max.pt"
    data.enc_min = f"./data/{data.dataset}/encodings-{model.hg_name_hash}-min.pt"

    config.decoder_path = f"./checkpoints/decoder-{config.model.hg_name_hash}-{config.data.dataset}.pth"
    config.seed = 0
    config.wandb = False
    config.ddp = True
    config.bert_config = bert_config
    config.project_name = '' #to be defined
    config.use_class = True
    config.num_cls = 0
    return config

bert_config = BertConfig(**{
    "hidden_size": 480,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
    "vocab_size": 30522,
    "hidden_dropout_prob": 0.1,
    "num_attention_heads": 16, #16
    "type_vocab_size": 2,
    "max_position_embeddings": 512,
    "num_hidden_layers": 12,
    "intermediate_size": 3072,
    "attention_probs_dropout_prob": 0.1,
    "layer_norm_eps": 1e-12,
    "model_type": "bert",
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "transformers_version": "4.6.0.dev0",
    "is_decoder": False,
})
