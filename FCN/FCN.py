from FCN.Pytorch_model_factory import Model_Factory_semantic_seg
from FCN.Pytorch_data_generator import Dataset_Generators


# Train the network
def fully_convolutional_wind_pred(cf):

    print(' ---> Init experiment: ' + cf.model_name + ' <---')
    # Create the data generators
    #show_DG(DG, 'train')  # this script will draw an image
    print('---> Building model...')
    model = Model_Factory_semantic_seg(cf)
    print('---> Creating Data generators...')
    DG = Dataset_Generators(cf)

    if cf.train_model:
        for epoch in range(1, cf.n_epochs + 1):
            model.train(DG.dataloader['train'], epoch)
            if epoch % cf.test_epoch == 0:
                if cf.test_model:
                    model.test(DG.dataloader['valid'], epoch, cf)

    print(' ---> Finish experiment: ' + cf.exp_name + ' <---')
