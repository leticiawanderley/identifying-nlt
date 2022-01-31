def create_hyperparam_files(learning_rates, numbers_hidden,
                            test_columns, losses, batch_sizes):
    file = 1
    for lr in learning_rates:
        for num in numbers_hidden:
            for col in test_columns:
                for loss in losses:
                    for size in batch_sizes:
                        f = open('hyperparams/input.' + str(file), 'w')
                        f.write(str(lr) + '\n' + str(num) + '\n' + \
                                col + '\n' + loss + '\n' + str(size))
                        f.close()
                        file += 1


create_hyperparam_files([0.01, 0.001, 0.0001, 0.00001, 0.000001],
                        [8, 16, 32, 64, 128, 256, 512],
                        ['incorrect_ud_tags_padded'],
                        ['BCEwithLL', 'NLLoss'],
                        [1, 2, 4, 8, 16, 32])
