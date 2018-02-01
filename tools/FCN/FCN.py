from FCN.Pytorch_model_factory import Model_Factory_semantic_seg

from tools.FCN.Pytorch_data_generator import Dataset_Generators, Dataset_Generators_no_crop


# Train the network
def fully_convolutional_wind_pred(cf):

    print('---> Building model...')
    model = Model_Factory_semantic_seg(cf)
    print('---> Creating Data generators...')
    DG = Dataset_Generators(cf)

    if cf.collect_train_valid_mse_iou:
        print('collect_train_valid_mse_iou')
        mse, iou = model.collect_train_valid_mse_iou(Dataset_Generators_no_crop(cf))
        print('mse: %.4f, iou: %.4f' % (mse, iou))
        # mse: 11.1299, iou: 0.5636

    if cf.train_model:
        for epoch in range(1, cf.n_epochs + 1):
            lr, loss = model.train(DG.dataloader['train'], epoch)
            if epoch % cf.valid_epoch == 0:
                if cf.valid_model:
                    print('epoch: %d, loss: %.4f.' % (epoch, loss))
                    model.test(DG.dataloader['valid'], epoch, cf)

    print(' ---> Finish experiment.')
