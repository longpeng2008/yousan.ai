import mxnet as mx
import logging
import os
import time

def _load_model(args, rank=0):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None)
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)

def _save_model(args, rank=0):
    dst_dir = os.path.dirname(args.model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(args.model_prefix if rank == 0 else "%s-%d" % (
        args.model_prefix, rank))

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--network', type=str,
                       help='the neural network to use')
    train.add_argument('--num-layers', type=int,
                       help='number of layers in the neural network, required by some networks such as resnet')
    train.add_argument('--gpus', type=str,
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    train.add_argument('--kv-store', type=str, default='device',
                       help='key-value store type')
    train.add_argument('--num-epochs', type=int, default=100,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate')
    train.add_argument('--optimizer', type=str, default='adam',
                       help='the optimizer type')
    train.add_argument('--wd', type=float, default=0.0001,
                       help='weight decay for sgd')
    train.add_argument('--batch-size', type=int, default=128,
                       help='the batch size')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix', type=str,
                       help='model prefix',default='simple-conv3')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    train.add_argument('--load-epoch', type=int,
                       help='load the model on an epoch using the model-load-prefix')
    train.add_argument('--test-io', type=int, default=0,
                       help='1 means test reading speed without training')
    return train

def fit(args, network, data_loader, **kwargs):
    """
    train a model
    args : argparse returns
    network : the symbol definition of the nerual network
    data_loader : function that returns the train and val data iterators
    """
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    # data iterators
    (train, val) = data_loader(args, kv)

    # load model
    if 'arg_params' in kwargs and 'aux_params' in kwargs:
        arg_params = kwargs['arg_params']
        aux_params = kwargs['aux_params']
    else:
        sym, arg_params, aux_params = _load_model(args, kv.rank)
        if sym is not None:
            assert sym.tojson() == network.tojson()

    # save model
    checkpoint = _save_model(args, kv.rank)

    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    lr = args.lr

    # create model
    model = mx.mod.Module(
        context       = devs,
        symbol        = network
    )

    #optimizer_params = {
     #       'learning_rate': lr,
      #      'momentum' : args.mom,
       #     'wd' : args.wd,
        #    'lr_scheduler': lr_scheduler}
    
    optimizer_params = {
            'learning_rate': lr,
            'wd' : args.wd}

    monitor = mx.mon.Monitor(args.monitor, pattern=".*") if args.monitor > 0 else None

    initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2.34)

    # evaluation metrices
    eval_metrics = ['accuracy']

    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]
    if 'batch_end_callback' in kwargs:
        cbs = kwargs['batch_end_callback']
        batch_end_callbacks += cbs if isinstance(cbs, list) else [cbs]

    # run
    model.fit(train,
        begin_epoch        = args.load_epoch if args.load_epoch else 0,
        num_epoch          = args.num_epochs,
        eval_data          = val,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        optimizer          = args.optimizer,
        optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        batch_end_callback = batch_end_callbacks,
        epoch_end_callback = checkpoint,
        allow_missing      = True,
        monitor            = monitor)
