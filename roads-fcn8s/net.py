import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, split='train', use_bn=False):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    
    relu = L.ReLU(conv, in_place=True)
    
    bn = None
    scale = None
    if use_bn:
        use_global_stats = True
        if 'train' in split:
            use_global_stats = False
        bn = L.BatchNorm(relu, use_global_stats=use_global_stats)
        scale = L.Scale(bn, scale_param=dict(bias_term=True))
    
    return conv, relu, bn, scale

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def fuse_scores(pool, rel_1, rel_2, rel_3, num_classes, lr_mult_score=1, decay_mult_score=1):

    pool_score = L.Convolution(pool, 
        num_output=num_classes, 
        kernel_size=1, 
        pad=0, 
        weight_filler=dict(type='xavier'),
        param=[dict(lr_mult=lr_mult_score, decay_mult=decay_mult_score), dict(lr_mult=2*lr_mult_score, decay_mult=0)])

    rel_1_score = L.Convolution(rel_1, 
        num_output=num_classes, 
        kernel_size=1, 
        pad=0, 
        weight_filler=dict(type='xavier'),
        param=[dict(lr_mult=lr_mult_score, decay_mult=decay_mult_score), dict(lr_mult=2*lr_mult_score, decay_mult=0)])

    rel_2_score = L.Convolution(rel_2, 
        num_output=num_classes, 
        kernel_size=1, 
        pad=0, 
        weight_filler=dict(type='xavier'),
        param=[dict(lr_mult=lr_mult_score, decay_mult=decay_mult_score), dict(lr_mult=2*lr_mult_score, decay_mult=0)])

    rel_3_score = L.Convolution(rel_3, 
        num_output=num_classes, 
        kernel_size=1, 
        pad=0, 
        weight_filler=dict(type='xavier'),
        param=[dict(lr_mult=lr_mult_score, decay_mult=decay_mult_score), dict(lr_mult=2*lr_mult_score, decay_mult=0)])

    fuse = L.Eltwise(pool_score, rel_1_score, rel_2_score, rel_3_score, operation=P.Eltwise.SUM)

    return pool_score, rel_1_score, rel_2_score, rel_3_score, fuse



def fcn_roads(split, batch_size):
    
    n = caffe.NetSpec()
    
    pydata_params = dict(split=split, mean=(103.07961905, 105.50374829, 99.71291237), voc_dir='../data/roads/ROADS/', 
                         batch_size=batch_size)
    pylayer = 'RoadsDataLayer'
    
    num_classes = 4;
        
    n.data, n.label = L.Python(module='roads_layers', layer=pylayer,
            ntop=2, param_str=str(pydata_params))

    # the base net
    n.conv1_1, n.relu1_1, _, _ = conv_relu(n.data, 64, pad=100, split=split)
    n.conv1_2, n.relu1_2, _, _ = conv_relu(n.relu1_1, 64, split=split)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1, _, _ = conv_relu(n.pool1, 128, split=split)
    n.conv2_2, n.relu2_2, _, _ = conv_relu(n.relu2_1, 128, split=split)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1, _, _ = conv_relu(n.pool2, 256, split=split)
    n.conv3_2, n.relu3_2, _, _ = conv_relu(n.relu3_1, 256, split=split)
    n.conv3_3, n.relu3_3, _, _ = conv_relu(n.relu3_2, 256, split=split)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1, _, _ = conv_relu(n.pool3, 512, split=split)
    n.conv4_2, n.relu4_2, _, _ = conv_relu(n.relu4_1, 512, split=split)
    n.conv4_3, n.relu4_3, _, _ = conv_relu(n.relu4_2, 512, split=split)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1, _, _ = conv_relu(n.pool4, 512, split=split)
    n.conv5_2, n.relu5_2, _, _ = conv_relu(n.relu5_1, 512, split=split)
    n.conv5_3, n.relu5_3, _, _ = conv_relu(n.relu5_2, 512, split=split)
    n.pool5 = max_pool(n.relu5_3)

    # fully conv
    n.fc6, n.relu6, _, _ = conv_relu(n.pool5, 4096, ks=7, pad=0, split=split)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.2, in_place=True)
    n.fc7, n.relu7, _, _ = conv_relu(n.drop6, 4096, ks=1, pad=0, split=split)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.2, in_place=True)
    
    # score

    lr_mult_score = 1
    decay_mult_score = 1

    n.score_fr_roads = L.Convolution(n.drop7, 
        num_output=num_classes, 
        kernel_size=1, 
        pad=0, 
        weight_filler=dict(type='xavier'),
        param=[dict(lr_mult=lr_mult_score, decay_mult=decay_mult_score), dict(lr_mult=2*lr_mult_score, decay_mult=0)])
    
    # n.score_fr_roads_2 = L.Convolution(n.pool5, 
    #     num_output=num_classes, 
    #     kernel_size=1, 
    #     pad=0, 
    #     weight_filler=dict(type='xavier'),
    #     param=[dict(lr_mult=lr_mult_score, decay_mult=decay_mult_score), dict(lr_mult=2*lr_mult_score, decay_mult=0)])
    
    
    # n.score_fr_roads = L.Eltwise(n.score_fr_roads_1, n.score_fr_roads_2, operation=P.Eltwise.SUM)
    
    
    n.upscore2_roads = L.Deconvolution(n.score_fr_roads,
        convolution_param=dict(num_output=num_classes, kernel_size=4, stride=2, bias_term=False),
        param=[dict(lr_mult=0)])
    
    # pool4 skip
    
    n.score_4p, n.score_51, n.score_52, n.score_53, n.score_pool4_roads = fuse_scores(n.pool4, 
        n.relu5_1, 
        n.relu5_2, 
        n.relu5_3, 
        num_classes)
    
    n.score_pool4c_roads = crop(n.score_pool4_roads, n.upscore2_roads)
    
    n.fuse_pool4_roads = L.Eltwise(n.upscore2_roads, n.score_pool4c_roads,
            operation=P.Eltwise.SUM)
    
    n.upscore_pool4_roads = L.Deconvolution(n.fuse_pool4_roads,
        convolution_param=dict(num_output=num_classes, kernel_size=4, stride=2, bias_term=False),
        param=[dict(lr_mult=0)])
        
    # pool3 skip

    n.score_3p, n.score_41, n.score_42, n.score_43, n.score_pool3_roads = fuse_scores(n.pool3, 
        n.relu4_1, 
        n.relu4_2, 
        n.relu4_3, 
        num_classes)
    
    n.score_pool3c_roads = crop(n.score_pool3_roads, n.upscore_pool4_roads)
    
    n.fuse_pool3_roads = L.Eltwise(n.upscore_pool4_roads, n.score_pool3c_roads,
            operation=P.Eltwise.SUM)

    n.upscore_pool3_roads = L.Deconvolution(n.fuse_pool3_roads,
        convolution_param=dict(num_output=num_classes, kernel_size=4, stride=2,
            bias_term=False),
        param=[dict(lr_mult=0)])

    # pool2 skip

    n.score_2p, n.score_31, n.score_32, n.score_33, n.score_pool2_roads = fuse_scores(n.pool2, 
        n.relu3_1, 
        n.relu3_2, 
        n.relu3_3, 
        num_classes)

    n.score_pool2c_roads = crop(n.score_pool2_roads, n.upscore_pool3_roads)

    n.fuse_pool2_roads = L.Eltwise(n.upscore_pool3_roads, n.score_pool2c_roads,
            operation=P.Eltwise.SUM)

    n.upscore4_roads = L.Deconvolution(n.fuse_pool2_roads,
        convolution_param=dict(num_output=num_classes, kernel_size=8, stride=4,
            bias_term=False),
        param=[dict(lr_mult=0)])

    # final score
    n.score = crop(n.upscore4_roads, n.data)
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
            loss_param=dict(normalize=False, ignore_label=255))

    return n.to_proto()


def make_nets():
    
        
    with open('train_roads.prototxt', 'w') as f:
        f.write(str(fcn_roads('train', 10)))

    with open('val_roads.prototxt', 'w') as f:
        f.write(str(fcn_roads('val', 1)))

if __name__ == '__main__':
    make_nets()
