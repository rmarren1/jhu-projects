def hyper_search(M, cp): 
    ACCS = []
    # load required dependencies
    import util.setup as setup
    import numpy as np
    from util import clean_data as cd
    import algos.basic as basic
    import algos.similarity as sim
    import experiments.static as static
    import util.viz as viz
    METADATA_PATH = '../data/Phenotypic_V1_0b_preprocessed1.csv'
    PREFIX = '../data/rois_ez/'
    POSTFIX = '_rois_ez.1D'
    # get meta-data of subjects
    data, labels, num_patients = setup.get_data(METADATA_PATH, PREFIX, POSTFIX)

    # mean center all of the data
    data = cd.run_function(basic.mean_center, data)

    # change from samples x dimensions to dimensions x samples
    data = cd.run_function(basic.transpose, data)

    # split subjects into training, testing, tuning (randomly)
    n_train = 684
    n_tune = 100
    n_test = 100
    for curr_params in cp:
        print 'Evaluating: ', curr_params
        p = np.random.permutation(len(data))
        data = [data[i] for i in p]
        labels = np.array(labels)[p]
        D, L = setup.split_data(data,
                                labels,
                                n_train,
                                n_tune,
                                n_test)
        # specify the model to use
        model = M.copy()
        model.update(**curr_params)
        mean_a = static.clust_corr(D['train']['a'], model)
        mean_c = static.clust_corr(D['train']['c'], model)
        diff = mean_a - mean_c
        bl = static.block_rep(D['tune'], model)
        H = model['thresh']
        accuracies = []
        thresholds = []
        for low in H:
            for high in H[H > low]:
                threshold = (low, high)
                acc, _, _ = static.thresh_simple(bl,
                                           L['tune'],
                                           diff,
                                           threshold,
                                           False)
                accuracies.append(acc)
                thresholds.append(threshold)
        best_tune_acc = np.max(accuracies)
        best_thresh = thresholds[np.argmax(accuracies)]
        bl = static.block_rep(D['tune'], model)
        acc, prec, rec = static.thresh_simple(bl,
                                   L['tune'],
                                   diff,
                                   best_thresh,
                                   True)
        ACCS.append((acc, p))
        print 'Accuracy: ', acc
    return ACCS