import torch


def generate_dict_vec(embedding_type, embedding_dim, pre_embedding_path, ):
    if embedding_type == 0:
        d_word_index, embed = torch.load(pre_embedding_path)
        print('load existing embedding vectors, name is ', pre_embedding_path)
    elif embedding_type == 1:
        v_builder = VocabBuilder(path_file=train_path)
        d_word_index, embed = v_builder.get_word_index(min_sample=args.min_samples)
        print('create new embedding vectors, training from scratch')
    elif embedding_type == 2:
        v_builder = VocabBuilder(path_file=train_path)
        d_word_index, embed = v_builder.get_word_index(min_sample=args.min_samples)
        embed = torch.randn([len(d_word_index), embedding_dim])
        print('create new embedding vectors, training the random vectors')
    else:
        raise ValueError('unsupported type')
    if embed is not None:
        assert embed.size()[1] == embedding_dim
